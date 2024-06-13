import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


PLOT_VALUE = True

if PLOT_VALUE:
    model = torch.load('neural-network-3-actor.pth')
else:
    model = torch.load('neural-network-3-critic.pth')

# Create a 3d plot for s(y, w) = (0, y, 0, 0, ω, 0, 0, 0) with w ∈ [−pi, pi] and y ∈ [0, 1.5] of
# V(s(y, w))


y = torch.linspace(0, 1.5, 100)
w = torch.linspace(-3.14, 3.14, 100)
y, w = torch.meshgrid(y, w)
s = torch.zeros((100, 100, 8))
s[:, :, 1] = y
s[:, :, 4] = w
s = s.reshape(-1, 8)
if PLOT_VALUE:
    mean, var = model(s)
    v_values = mean[:, 1]
else:
    v_values = model(s)
v_values = v_values.reshape(100, 100)
v_values = v_values.detach().numpy()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(y, w, v_values)
ax.set_xlabel('y')
ax.set_ylabel('ω')
ax.set_zlabel('μ(s)[:, 1]')
ax.set_title('μ(s)[:, 1]')

plt.show()