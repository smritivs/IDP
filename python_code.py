import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import qmc
from PIL import Image
import cv2
from skimage import measure

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

# Image processing functions
def load_and_process_image(image_path, target_width=600, target_height=300):
    """
    Load image and process it to get geometry information
    White pixels = solid obstacles (cylinders/walls)
    Black pixels = fluid domain
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('L')  
    else:
        if isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)
        else:
            img = image_path
        img = img.convert('L')

    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
   
    img_array = np.array(img) / 255.0
    
    # Threshold: >0.5 = solid (white), <0.5 = fluid (black)
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
    
    # Clip to image bounds
    height, width = img_shape
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, height - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, width - 1)
    
    # Convert to integer indices
    row_idx = pixel_coords[:, 0].astype(int)
    col_idx = pixel_coords[:, 1].astype(int)
    
    # Check if points are in fluid domain (not solid)
    return ~solid_mask[row_idx, col_idx]

def extract_boundary_points_from_image(solid_mask, domain_bounds, n_boundary_points=500):
    """Extract boundary points between solid and fluid regions"""
    # Find contours of solid regions
    contours = measure.find_contours(solid_mask.astype(float), 0.5)
    
    all_boundary_points = []
    
    for contour in contours:
        if len(contour) > 10:  # Filter out very small contours
            # Convert contour points to physical coordinates
            boundary_points = image_to_physical_coords(contour, solid_mask.shape, domain_bounds)
            all_boundary_points.extend(boundary_points)
    
    all_boundary_points = np.array(all_boundary_points)
    
    # Subsample if we have too many points
    if len(all_boundary_points) > n_boundary_points:
        indices = np.linspace(0, len(all_boundary_points) - 1, n_boundary_points, dtype=int)
        all_boundary_points = all_boundary_points[indices]
    
    return all_boundary_points

# PINN Model (same as before)
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

# Modified point generation functions
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
    
    # Left boundary (inlet): x = x_min
    left = np.column_stack([np.full(n_per_edge, domain_bounds[0][0]), 
                           np.linspace(domain_bounds[1][0], domain_bounds[1][1], n_per_edge)])
    
    # Right boundary (outlet): x = x_max
    right = np.column_stack([np.full(n_per_edge, domain_bounds[0][1]),
                            np.linspace(domain_bounds[1][0], domain_bounds[1][1], n_per_edge)])
    
    # Bottom boundary: y = y_min
    bottom = np.column_stack([np.linspace(domain_bounds[0][0], domain_bounds[0][1], n_per_edge),
                             np.full(n_per_edge, domain_bounds[1][0])])
    
    # Top boundary: y = y_max
    top = np.column_stack([np.linspace(domain_bounds[0][0], domain_bounds[0][1], n_per_edge),
                          np.full(n_per_edge, domain_bounds[1][1])])
    
    return np.vstack([left, right, bottom, top])

def generate_fluid_domain_points_from_image(n_points, domain_bounds, solid_mask, img_shape):
    """Generate collocation points in fluid domain based on image mask"""
    max_attempts = n_points * 10
    points = []
    attempts = 0
    
    while len(points) < n_points and attempts < max_attempts:
        # Generate candidate points using LHS
        batch_size = min(1000, (n_points - len(points)) * 2)
        candidate_points = generate_lhs_points(batch_size, domain_bounds)
        
        # Filter points to keep only those in fluid domain
        in_fluid = is_in_fluid_domain(candidate_points, solid_mask, img_shape, domain_bounds)
        valid_points = candidate_points[in_fluid]
        
        points.extend(valid_points)
        attempts += batch_size
    
    print(f"Generated {len(points[:n_points])} fluid points after {attempts} attempts")
    return np.array(points[:n_points])

# Physics and training functions (mostly same as before)
def navier_stokes_residual(xy, model, nu=0.02, rho=1.0):
    """Compute Navier-Stokes equation residuals using automatic differentiation"""
    xy.requires_grad_(True)
    uvp = model(xy)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    
    # First derivatives
    grads_u = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    grads_p = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]
    
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, xy, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]  
    v_yy = torch.autograd.grad(v_y, xy, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    
    # Navier-Stokes equations
    res_continuity = u_x + v_y
    res_momentum_x = u * u_x + v * u_y + p_x/rho - nu * (u_xx + u_yy)
    res_momentum_y = u * v_x + v * v_y + p_y/rho - nu * (v_xx + v_yy)
    
    return res_continuity, res_momentum_x, res_momentum_y

def train_pinn_with_image(model, fluid_points, boundary_points, obstacle_boundary_points,
                         domain_bounds, inlet_velocity=1.0, epochs=3000, lr=1e-3):
    """Train PINN model with image-based geometry"""
    
    # Convert to tensors
    fluid_tensor = torch.tensor(fluid_points, dtype=torch.float32, device=device)
    boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    obstacle_tensor = torch.tensor(obstacle_boundary_points, dtype=torch.float32, device=device)
    
    # Identify boundary types based on domain bounds
    left_mask = boundary_points[:, 0] < (domain_bounds[0][0] + 0.01)   # Inlet
    right_mask = boundary_points[:, 0] > (domain_bounds[0][1] - 0.01)  # Outlet
    wall_mask = ~(left_mask | right_mask)     # Top/bottom walls
    
    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    print("Training PINN for image-based geometry flow...")
    print(f"Fluid points: {len(fluid_points)}")
    print(f"Domain boundary points: {len(boundary_points)}")
    print(f"Obstacle boundary points: {len(obstacle_boundary_points)}")
    print()
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics loss
        res_c, res_mx, res_my = navier_stokes_residual(fluid_tensor, model, nu=0.02, rho=1.0)
        physics_loss = (res_c**2 + res_mx**2 + res_my**2).mean()
        
        # Domain boundary losses
        boundary_loss = 0.0
        
        # Inlet boundary condition: u = inlet_velocity, v = 0
        if np.any(left_mask):
            inlet_tensor = boundary_tensor[left_mask]
            inlet_output = model(inlet_tensor)
            inlet_loss = ((inlet_output[:, 0] - inlet_velocity)**2 + 
                         (inlet_output[:, 1])**2).mean()
            boundary_loss += 10.0 * inlet_loss
        
        # Outlet boundary condition: pressure outlet (p = 0)
        if np.any(right_mask):
            outlet_tensor = boundary_tensor[right_mask]
            outlet_output = model(outlet_tensor)
            outlet_loss = (outlet_output[:, 2]**2).mean()
            boundary_loss += 1.0 * outlet_loss
        
        # Wall boundaries: u = v = 0 (slip condition for top/bottom walls)
        if np.any(wall_mask):
            wall_tensor = boundary_tensor[wall_mask]
            wall_output = model(wall_tensor)
            wall_loss = (wall_output[:, :2]**2).mean()
            boundary_loss += 5.0 * wall_loss
        
        # Obstacle boundary: no-slip condition (u = v = 0)
        if len(obstacle_boundary_points) > 0:
            obstacle_output = model(obstacle_tensor)
            obstacle_loss = (obstacle_output[:, :2]**2).mean()
        else:
            obstacle_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = physics_loss + boundary_loss + 100.0 * obstacle_loss
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Physics: {physics_loss.item():.6f}")
            print(f"  Boundary: {boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss:.6f}")
            print(f"  Obstacle: {obstacle_loss.item():.6f}")
            print()
    
    return loss_history

def plot_results_with_image(model, solid_mask, img_shape, domain_bounds, resolution=600):
    """Plot velocity components, pressure, and streamlines with image-based geometry"""
    
    x = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution//2)
    X, Y = np.meshgrid(x, y)
    
    coords = np.column_stack([X.ravel(), Y.ravel()])
    
    # Filter out points in solid regions
    in_fluid = is_in_fluid_domain(coords, solid_mask, img_shape, domain_bounds)
    fluid_coords = coords[in_fluid]
    
    # Evaluate model only on fluid points
    coords_tensor = torch.tensor(fluid_coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        output = model(coords_tensor)
    
    # Create full arrays filled with NaN
    u_full = np.full(len(coords), np.nan)
    v_full = np.full(len(coords), np.nan)
    p_full = np.full(len(coords), np.nan)
    
    # Fill fluid regions with predicted values
    u_full[in_fluid] = output[:, 0].cpu().numpy()
    v_full[in_fluid] = output[:, 1].cpu().numpy()
    p_full[in_fluid] = output[:, 2].cpu().numpy()
    
    # Reshape to grid
    u = u_full.reshape(resolution//2, resolution)
    v = v_full.reshape(resolution//2, resolution)
    p = p_full.reshape(resolution//2, resolution)
    speed = np.sqrt(u**2 + v**2)
    
    # Create masks
    u_masked = np.ma.masked_where(np.isnan(u), u)
    v_masked = np.ma.masked_where(np.isnan(v), v)
    p_masked = np.ma.masked_where(np.isnan(p), p)
    speed_masked = np.ma.masked_where(np.isnan(speed), speed)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
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
    
    # Add streamlines (skip NaN regions)
    step = max(1, resolution // 20)
    X_stream = X[::step, ::step]
    Y_stream = Y[::step, ::step]
    U_stream = u[::step, ::step]
    V_stream = v[::step, ::step]
    
    # Mask streamline data
    mask_stream = np.isnan(U_stream) | np.isnan(V_stream)
    U_stream = np.ma.masked_where(mask_stream, U_stream)
    V_stream = np.ma.masked_where(mask_stream, V_stream)
    
    try:
        axes[1,1].streamplot(X_stream, Y_stream, U_stream, V_stream, 
                            density=1.5, color='white', linewidth=1, arrowsize=1.5)
    except:
        print("Warning: Could not generate streamlines")
    
    axes[1,1].set_title('Speed with Streamlines')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1,1])
    
    # Overlay solid regions
    for ax in axes.flat:
        # Convert solid mask to physical coordinates for overlay
        y_img = np.linspace(domain_bounds[1][1], domain_bounds[1][0], img_shape[0])
        x_img = np.linspace(domain_bounds[0][0], domain_bounds[0][1], img_shape[1])
        X_img, Y_img = np.meshgrid(x_img, y_img)
        
        ax.contour(X_img, Y_img, solid_mask, levels=[0.5], colors='black', linewidths=2)
        ax.contourf(X_img, Y_img, solid_mask, levels=[0.5, 1.0], colors=['black'], alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return u_masked, v_masked, p_masked, speed_masked

def plot_loss_history(loss_history):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss')
    plt.grid(True)
    plt.show()

# Import the custom CUDA extension (assume it's built and named 'fourier_cuda')
try:
    import fourier_cuda  # This should be the name of your compiled extension
except ImportError:
    fourier_cuda = None
    print("[WARNING] fourier_cuda extension not found. Using fallback.")

def fourier_transform(x, freq=None):
    """Standalone demonstration of a custom fourier kernel (calls CUDA if available)"""
    print("[DEMO] fourier_transform called with shape:", x.shape)
    if fourier_cuda is not None:
        # Ensure input is a torch tensor on CUDA
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().cuda()
        else:
            x = x.cuda()
        if freq is None:
            raise ValueError("Frequency matrix 'freq' must be provided for CUDA kernel.")
        if not isinstance(freq, torch.Tensor):
            freq = torch.from_numpy(freq).float().cuda()
        else:
            freq = freq.cuda()
        # Call the custom CUDA kernel (replace 'fourier_forward' with your function name)
        result = fourier_cuda.fourier_forward(x, freq)
        return result.cpu().numpy()
    else:
        # Fallback: use numpy FFT for demonstration
        return np.fft.fft(x, axis=0)

# Main function for image-based geometry
def run_image_based_flow_case(image_path, domain_bounds=[[0, 4], [0, 2]], 
                             inlet_velocity=1.0, epochs=3000, 
                             n_fluid_points=10000, n_boundary_points=600):
    """
    Run PINN flow analysis for geometry defined by an image
    
    Parameters:
    - image_path: Path to image file (white = solid, black = fluid)
    - domain_bounds: Physical domain bounds [[x_min, x_max], [y_min, y_max]]
    - inlet_velocity: Inlet velocity
    - epochs: Training epochs
    - n_fluid_points: Number of collocation points in fluid domain
    - n_boundary_points: Number of boundary points
    """
    print("=== PINN for Image-Based Geometry Flow ===")
    print("White regions in image = solid obstacles")
    print("Black regions in image = fluid domain")
    print()
    
    # Load and process image
    print("Loading and processing image...")
    img_array, solid_mask, fluid_mask = load_and_process_image(image_path)
    img_shape = solid_mask.shape
    
    # Display the processed image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray', origin='upper')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(solid_mask, cmap='RdYlBu', origin='upper')
    plt.title('Processed Geometry\n(Blue=Fluid, Red=Solid)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Image shape: {img_shape}")
    print(f"Solid area fraction: {np.sum(solid_mask) / solid_mask.size:.3f}")
    print()
    
    # Generate point clouds
    print("Generating point clouds...")
    
    # Fluid domain points
    fluid_points = generate_fluid_domain_points_from_image(
        n_fluid_points, domain_bounds, solid_mask, img_shape)
    
    # Domain boundary points
    boundary_points = generate_domain_boundary_points(n_boundary_points, domain_bounds)
    
    # Obstacle boundary points from image
    obstacle_boundary_points = extract_boundary_points_from_image(
        solid_mask, domain_bounds, n_boundary_points//2)
    
    print(f"Generated {len(fluid_points)} fluid domain points")
    print(f"Generated {len(boundary_points)} domain boundary points")
    print(f"Generated {len(obstacle_boundary_points)} obstacle boundary points")
    print()
    
    # Create PINN model
    model = PINN(hidden_layers=20, hidden_units=50).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created PINN with {total_params} parameters")
    print()
     
    # Train the model
    loss_history = train_pinn_with_image(
        model, fluid_points, boundary_points, obstacle_boundary_points,
        domain_bounds, inlet_velocity=inlet_velocity, epochs=epochs)
    
    # Plot results
    print("Generating plots...")
    plot_loss_history(loss_history)
    u, v, p, speed = plot_results_with_image(model, solid_mask, img_shape, domain_bounds)
    
    return model, loss_history, u, v, p, speed, solid_mask

if __name__ == "__main__":
    
    # Test Image
    # print("Creating test image with circular obstacle...")
    # test_img = np.zeros((200, 400))
    # center_x, center_y = 100, 100  # pixel coordinates
    # radius = 15
    # y_coords, x_coords = np.ogrid[:200, :400]
    # mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
    # test_img[mask] = 255 
    
    
    # test_image = Image.fromarray(test_img.astype(np.uint8))
    # test_image.save("test_cylinder.png")
    
    
    results = run_image_based_flow_case(
        "NACA0012.png", 
        domain_bounds=[[0, 6], [0, 3]], 
        inlet_velocity=1.0, 
        epochs=1000 
    )
    
    if results:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed")

    # Standalone demonstration of fourier_transform using actual fluid_points (regenerated for demo)
    print("[DEMO] Running fourier_transform on regenerated fluid_points...")
    # Regenerate a small set of fluid points for demonstration
    demo_domain_bounds = [[0, 6], [0, 3]]
    demo_n_points = 16
    demo_fluid_points = generate_lhs_points(demo_n_points, demo_domain_bounds).astype(np.float32)
    demo_freq = np.random.rand(2, 4).astype(np.float32)  # Example frequency matrix
    fourier_result = fourier_transform(demo_fluid_points, freq=demo_freq)
    print("[DEMO] Fourier transform result shape:", fourier_result.shape)
    print("[DEMO] Fourier transform result (first 2 rows):\n", fourier_result[:2])