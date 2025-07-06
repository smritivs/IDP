from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import qmc
from PIL import Image
from skimage import measure
import tempfile
import uuid
import traceback
import json
import time

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Import your PINN code functions
# (I'll include the necessary functions here, but you can also import from your module)

# Device setup
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_device()

# PINN Model
class PINN(nn.Module):
    def __init__(self, hidden_layers=20, hidden_units=50):
        super().__init__()
        
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        
        layers += [nn.Linear(hidden_units, 3)]
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# Image processing functions
def load_and_process_image(image_input, target_width=600, target_height=300):
    """Load image and process it to get geometry information"""
    if isinstance(image_input, str):
        # File path
        img = Image.open(image_input).convert('L')  
    elif hasattr(image_input, 'read'):
        # File-like object (uploaded file)
        img = Image.open(image_input).convert('L')
    elif hasattr(image_input, 'stream'):
        # Flask FileStorage object
        img = Image.open(image_input.stream).convert('L')
    else:
        # NumPy array or PIL Image
        if isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = image_input
        img = img.convert('L')

    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    img_array = np.array(img) / 255.0
    solid_mask = img_array > 0.5
    fluid_mask = ~solid_mask
    
    return img_array, solid_mask, fluid_mask

def image_to_physical_coords(pixel_coords, img_shape, domain_bounds):
    """Convert pixel coordinates to physical coordinates"""
    height, width = img_shape
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    
    x_phys = pixel_coords[:, 1] * (x_max - x_min) / width + x_min
    y_phys = (height - 1 - pixel_coords[:, 0]) * (y_max - y_min) / height + y_min
    
    return np.column_stack([x_phys, y_phys])

def physical_to_image_coords(phys_coords, img_shape, domain_bounds):
    """Convert physical coordinates to pixel coordinates"""
    height, width = img_shape
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    
    pixel_x = (phys_coords[:, 0] - x_min) * width / (x_max - x_min)
    pixel_y = height - 1 - (phys_coords[:, 1] - y_min) * height / (y_max - y_min)
    
    return np.column_stack([pixel_y, pixel_x])

def is_in_fluid_domain(coords, solid_mask, img_shape, domain_bounds):
    """Check if physical coordinates are in fluid domain"""
    pixel_coords = physical_to_image_coords(coords, img_shape, domain_bounds)
    
    height, width = img_shape
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, height - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, width - 1)
    
    row_idx = pixel_coords[:, 0].astype(int)
    col_idx = pixel_coords[:, 1].astype(int)
    
    return ~solid_mask[row_idx, col_idx]

def extract_boundary_points_from_image(solid_mask, domain_bounds, n_boundary_points=500):
    """Extract boundary points between solid and fluid regions"""
    contours = measure.find_contours(solid_mask.astype(float), 0.5)
    
    all_boundary_points = []
    
    for contour in contours:
        if len(contour) > 10:
            boundary_points = image_to_physical_coords(contour, solid_mask.shape, domain_bounds)
            all_boundary_points.extend(boundary_points)
    
    all_boundary_points = np.array(all_boundary_points)
    
    if len(all_boundary_points) > n_boundary_points:
        indices = np.linspace(0, len(all_boundary_points) - 1, n_boundary_points, dtype=int)
        all_boundary_points = all_boundary_points[indices]
    
    return all_boundary_points

def generate_lhs_points(n_points, bounds):
    """Generate points using Latin Hypercube Sampling"""
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n_points)
    
    x_points = sample[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    y_points = sample[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
    
    return np.column_stack([x_points, y_points])

def generate_domain_boundary_points(n_points=600, domain_bounds=[[0, 4], [0, 2]]):
    """Generate points on domain boundaries for freestream conditions"""
    n_per_edge = n_points // 4
    
    left = np.column_stack([np.full(n_per_edge, domain_bounds[0][0]), 
                           np.linspace(domain_bounds[1][0], domain_bounds[1][1], n_per_edge)])
    
    right = np.column_stack([np.full(n_per_edge, domain_bounds[0][1]),
                            np.linspace(domain_bounds[1][0], domain_bounds[1][1], n_per_edge)])
    
    bottom = np.column_stack([np.linspace(domain_bounds[0][0], domain_bounds[0][1], n_per_edge),
                             np.full(n_per_edge, domain_bounds[1][0])])
    
    top = np.column_stack([np.linspace(domain_bounds[0][0], domain_bounds[0][1], n_per_edge),
                          np.full(n_per_edge, domain_bounds[1][1])])
    
    return np.vstack([left, right, bottom, top])

def generate_fluid_domain_points_from_image(n_points, domain_bounds, solid_mask, img_shape):
    """Generate collocation points in fluid domain based on image mask"""
    max_attempts = n_points * 10
    points = []
    attempts = 0
    
    while len(points) < n_points and attempts < max_attempts:
        batch_size = min(1000, (n_points - len(points)) * 2)
        candidate_points = generate_lhs_points(batch_size, domain_bounds)
        
        in_fluid = is_in_fluid_domain(candidate_points, solid_mask, img_shape, domain_bounds)
        valid_points = candidate_points[in_fluid]
        
        points.extend(valid_points)
        attempts += batch_size
    
    return np.array(points[:n_points])

def navier_stokes_residual(xy, model, nu=0.02, rho=1.0):
    """Compute Navier-Stokes equation residuals using automatic differentiation"""
    xy.requires_grad_(True)
    uvp = model(xy)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    
    grads_u = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    grads_p = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]
    
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]
    
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, xy, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]  
    v_yy = torch.autograd.grad(v_y, xy, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    
    res_continuity = u_x + v_y
    res_momentum_x = u * u_x + v * u_y + p_x/rho - nu * (u_xx + u_yy)
    res_momentum_y = u * v_x + v * v_y + p_y/rho - nu * (v_xx + v_yy)
    
    return res_continuity, res_momentum_x, res_momentum_y

def train_pinn_with_image(model, fluid_points, boundary_points, obstacle_boundary_points,
                         domain_bounds, inlet_velocity=1.0, epochs=3000, lr=1e-3):
    """Train PINN model with image-based geometry"""
    
    fluid_tensor = torch.tensor(fluid_points, dtype=torch.float32, device=device)
    boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    obstacle_tensor = torch.tensor(obstacle_boundary_points, dtype=torch.float32, device=device)
    
    left_mask = boundary_points[:, 0] < (domain_bounds[0][0] + 0.01)
    right_mask = boundary_points[:, 0] > (domain_bounds[0][1] - 0.01)
    wall_mask = ~(left_mask | right_mask)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        res_c, res_mx, res_my = navier_stokes_residual(fluid_tensor, model, nu=0.02, rho=1.0)
        physics_loss = (res_c**2 + res_mx**2 + res_my**2).mean()
        
        boundary_loss = 0.0
        
        if np.any(left_mask):
            inlet_tensor = boundary_tensor[left_mask]
            inlet_output = model(inlet_tensor)
            inlet_loss = ((inlet_output[:, 0] - inlet_velocity)**2 + 
                         (inlet_output[:, 1])**2).mean()
            boundary_loss += 10.0 * inlet_loss
        
        if np.any(right_mask):
            outlet_tensor = boundary_tensor[right_mask]
            outlet_output = model(outlet_tensor)
            outlet_loss = (outlet_output[:, 2]**2).mean()
            boundary_loss += 1.0 * outlet_loss
        
        if np.any(wall_mask):
            wall_tensor = boundary_tensor[wall_mask]
            wall_output = model(wall_tensor)
            wall_loss = (wall_output[:, :2]**2).mean()
            boundary_loss += 5.0 * wall_loss
        
        if len(obstacle_boundary_points) > 0:
            obstacle_output = model(obstacle_tensor)
            obstacle_loss = (obstacle_output[:, :2]**2).mean()
        else:
            obstacle_loss = torch.tensor(0.0, device=device)
        
        total_loss = physics_loss + boundary_loss + 100.0 * obstacle_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        loss_history.append(total_loss.item())
        
        # Print progress less frequently to avoid overwhelming logs
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6f}")
    
    return loss_history

def plot_results_with_image(model, solid_mask, img_shape, domain_bounds, resolution=400):
    """Plot velocity components, pressure, and streamlines with image-based geometry"""
    
    x = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution//2)
    X, Y = np.meshgrid(x, y)
    
    coords = np.column_stack([X.ravel(), Y.ravel()])
    
    in_fluid = is_in_fluid_domain(coords, solid_mask, img_shape, domain_bounds)
    fluid_coords = coords[in_fluid]
    
    coords_tensor = torch.tensor(fluid_coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        output = model(coords_tensor)
    
    u_full = np.full(len(coords), np.nan)
    v_full = np.full(len(coords), np.nan)
    p_full = np.full(len(coords), np.nan)
    
    u_full[in_fluid] = output[:, 0].cpu().numpy()
    v_full[in_fluid] = output[:, 1].cpu().numpy()
    p_full[in_fluid] = output[:, 2].cpu().numpy()
    
    u = u_full.reshape(resolution//2, resolution)
    v = v_full.reshape(resolution//2, resolution)
    p = p_full.reshape(resolution//2, resolution)
    speed = np.sqrt(u**2 + v**2)
    
    u_masked = np.ma.masked_where(np.isnan(u), u)
    v_masked = np.ma.masked_where(np.isnan(v), v)
    p_masked = np.ma.masked_where(np.isnan(p), p)
    speed_masked = np.ma.masked_where(np.isnan(speed), speed)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # U velocity
    im1 = axes[0,0].contourf(X, Y, u_masked, levels=20, cmap='RdBu_r')
    axes[0,0].set_title('u-velocity (horizontal)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0,0])
    
    # V velocity
    im2 = axes[0,1].contourf(X, Y, v_masked, levels=20, cmap='RdBu_r')
    axes[0,1].set_title('v-velocity (vertical)')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Pressure
    im3 = axes[1,0].contourf(X, Y, p_masked, levels=20, cmap='viridis')
    axes[1,0].set_title('Pressure')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Speed with streamlines
    im4 = axes[1,1].contourf(X, Y, speed_masked, levels=20, cmap='plasma')
    
    step = max(1, resolution // 20)
    X_stream = X[::step, ::step]
    Y_stream = Y[::step, ::step]
    U_stream = u[::step, ::step]
    V_stream = v[::step, ::step]
    
    mask_stream = np.isnan(U_stream) | np.isnan(V_stream)
    U_stream = np.ma.masked_where(mask_stream, U_stream)
    V_stream = np.ma.masked_where(mask_stream, V_stream)
    
    try:
        axes[1,1].streamplot(X_stream, Y_stream, U_stream, V_stream, 
                            density=1.5, color='white', linewidth=1, arrowsize=1.5)
    except:
        pass
    
    axes[1,1].set_title('Speed with Streamlines')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1,1])
    
    # Overlay solid regions
    for ax in axes.flat:
        y_img = np.linspace(domain_bounds[1][1], domain_bounds[1][0], img_shape[0])
        x_img = np.linspace(domain_bounds[0][0], domain_bounds[0][1], img_shape[1])
        X_img, Y_img = np.meshgrid(x_img, y_img)
        
        ax.contour(X_img, Y_img, solid_mask, levels=[0.5], colors='black', linewidths=2)
        ax.contourf(X_img, Y_img, solid_mask, levels=[0.5, 1.0], colors=['black'], alpha=0.3)
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    return buffer

def plot_geometry(img_array, solid_mask):
    """Plot the processed geometry"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_array, cmap='gray', origin='upper')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(solid_mask, cmap='RdYlBu', origin='upper')
    axes[1].set_title('Processed Geometry\n(Blue=Fluid, Red=Solid)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    return buffer

def plot_loss_history(loss_history):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss')
    plt.grid(True)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    return buffer

def analyze_flow_from_image(image_file, domain_bounds, inlet_velocity=1.0, epochs=1000, 
                          n_fluid_points=10000, n_boundary_points=600):
    """Main function to analyze flow from uploaded image"""
    
    # Process image - handle Flask FileStorage object
    img_array, solid_mask, fluid_mask = load_and_process_image(image_file)
    img_shape = solid_mask.shape
    
    # Generate point clouds
    fluid_points = generate_fluid_domain_points_from_image(
        n_fluid_points, domain_bounds, solid_mask, img_shape)
    
    boundary_points = generate_domain_boundary_points(n_boundary_points, domain_bounds)
    
    obstacle_boundary_points = extract_boundary_points_from_image(
        solid_mask, domain_bounds, n_boundary_points//2)
    
    # Create and train model
    model = PINN(hidden_layers=15, hidden_units=40).to(device)
    
    loss_history = train_pinn_with_image(
        model, fluid_points, boundary_points, obstacle_boundary_points,
        domain_bounds, inlet_velocity=inlet_velocity, epochs=epochs)
    
    # Generate plots
    geometry_plot = plot_geometry(img_array, solid_mask)
    loss_plot = plot_loss_history(loss_history)
    results_plot = plot_results_with_image(model, solid_mask, img_shape, domain_bounds)
    
    return geometry_plot, loss_plot, results_plot

# Global variable to store training progress
training_progress = []

def clear_progress():
    global training_progress
    training_progress = []

def add_progress(message):
    global training_progress
    training_progress.append(message)

@app.route('/progress')
def progress():
    def generate():
        global training_progress
        last_sent = 0
        while True:
            if len(training_progress) > last_sent:
                for i in range(last_sent, len(training_progress)):
                    yield f"data: {training_progress[i]}\n\n"
                last_sent = len(training_progress)
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/plain')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/test')
def test():
    return "Flask server is working!"

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        try:
            x_min = float(request.form.get('xMin', 0))
            x_max = float(request.form.get('xMax', 6))
            y_min = float(request.form.get('yMin', 0))
            y_max = float(request.form.get('yMax', 3))
            inlet_velocity = float(request.form.get('inletVelocity', 1.0))
            epochs = int(request.form.get('epochs', 1000))
            n_fluid_points = int(request.form.get('fluidPoints', 10000))
            n_boundary_points = int(request.form.get('boundaryPoints', 600))
        except ValueError as e:
            return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
        
        domain_bounds = [[x_min, x_max], [y_min, y_max]]
        
        # Validate parameters
        if x_min >= x_max or y_min >= y_max:
            return jsonify({'error': 'Invalid domain bounds'}), 400
        
        if epochs < 100 or epochs > 10000:
            return jsonify({'error': 'Epochs must be between 100 and 10000'}), 400
        
        if n_fluid_points < 1000 or n_fluid_points > 50000:
            return jsonify({'error': 'Fluid points must be between 1000 and 50000'}), 400
        
        print(f"Starting analysis with parameters:")
        print(f"  Domain bounds: {domain_bounds}")
        print(f"  Inlet velocity: {inlet_velocity}")
        print(f"  Epochs: {epochs}")
        print(f"  Fluid points: {n_fluid_points}")
        print(f"  Boundary points: {n_boundary_points}")
        
        # Clear previous progress
        clear_progress()
        
        # Run analysis
        geometry_plot, loss_plot, results_plot = analyze_flow_from_image(
            image_file, domain_bounds, inlet_velocity, epochs, 
            n_fluid_points, n_boundary_points)
        
        # Convert plots to base64
        geometry_b64 = base64.b64encode(geometry_plot.getvalue()).decode('utf-8')
        loss_b64 = base64.b64encode(loss_plot.getvalue()).decode('utf-8')
        results_b64 = base64.b64encode(results_plot.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'geometry_image': geometry_b64,
            'loss_plot': loss_b64,
            'flow_results': results_b64
        })
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    print("Starting PINN Flow Analysis Server...")
    print("Make sure to save the HTML frontend as 'static/index.html'")
    print("Server will be available at: http://localhost:8080")
    
    app.run(debug=True, host='127.0.0.1', port=8080)