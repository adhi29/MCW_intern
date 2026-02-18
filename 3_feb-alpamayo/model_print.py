from torchinfo import summary

# Create dummy inputs
dummy_images = torch.randn(1, 2, 3, 448, 280)  # batch=1, 2 cameras
dummy_ego = torch.randn(1, 10)  # ego-motion

# Print detailed summary
summary(
    model, 
    input_data=[dummy_images, dummy_ego],
    depth=5,  # How deep to show nested modules
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    row_settings=["var_names"],
    verbose=2
)