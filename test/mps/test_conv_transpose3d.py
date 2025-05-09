import torch
import itertools
import unittest
import functools

# Assuming PyTorch test infrastructure utilities are available in this context
# If not, we might need to adjust imports based on the exact environment
# (e.g., from torch.testing._internal.common_utils import TestCase, run_tests, skipIfMPS)
# For now, using standard unittest and manual MPS check.

# Placeholder for potential PyTorch testing utilities if needed directly
class TestCase(unittest.TestCase):
    # Add common test utilities if required later
    pass

def skipIfMPS(func):
    """Decorator to skip tests if MPS is not available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS not available")
        return func(*args, **kwargs)
    return wrapper

# If running within PyTorch's test suite, these might be implicitly available
# Otherwise, define basic versions or import them if possible.
try:
    # Attempt to import from PyTorch's internal test utilities
    from torch.testing._internal.common_utils import TestCase, run_tests, skipIfMPS, IS_MPS
    import torch.testing._internal.common_utils
except ImportError:
    # Fallback if internal utils are not directly accessible in this context
    IS_MPS = torch.backends.mps.is_available()
    # Basic run_tests functionality
    def run_tests():
        unittest.main()

class TestConvTranspose3dMPS(TestCase):

    @skipIfMPS
    def test_conv_transpose3d_configs(self):
        """Tests ConvTranspose3d on MPS with various stride/padding/dilation configs."""
        torch.manual_seed(0)
        # Configurations: (stride, padding, dilation)
        configs = list(itertools.product((1, 2), (0, 1), (1, 2)))

        for i, cfg in enumerate(configs):
            stride, pad, dil = cfg
            with self.subTest(f"Config_{i}", cfg=cfg):
                print(f"Testing config: {cfg}") # Optional: keep for verbose output
                try:
                    # Inputs
                    x = torch.randn(2, 3, 16, 16, 16, device='mps', requires_grad=True)
                    k = torch.randn(3, 3, 3, 3, 3, device='mps', requires_grad=True)

                    # Forward pass
                    y = torch.nn.functional.conv_transpose3d(
                        x, k, padding=(pad,) * 3, output_padding=(0,) * 3,
                        stride=(stride,) * 3, dilation=(dil,) * 3
                    )
                    # print(f"  Forward pass done. y shape: {y.shape}") # Optional

                    # Loss and Backward pass
                    loss = y.square().mean()
                    # print("  Loss calculated.") # Optional
                    loss.backward()
                    # print("  Backward pass done.") # Optional

                    # Check gradient
                    self.assertIsNotNone(k.grad, msg=f"Gradient is None for config {cfg}")
                    self.assertEqual(k.grad.shape, k.shape, msg=f"Gradient shape mismatch for config {cfg}")
                    # print(f"  Result: {cfg} grad OK {k.grad.shape}") # Optional

                except Exception as e:
                    self.fail(f"!!! Error during config {cfg}: {e}")

if __name__ == '__main__':
    run_tests() 