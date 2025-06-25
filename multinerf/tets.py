import jax, jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("available devices:", jax.devices())
print("ptxas:", __import__('subprocess').getoutput('ptxas --version'))
print("jaxlib:", jax.__version__)
print("devices:", jax.devices())