import jax
import numpy as np
from PIL import Image
import jax.numpy as jnp
import optax
from functools import partial
import fibonacci

_MAX_DEGREE = 20
_ITERATIONS = 5000
_NUM_POINTS = 500
WIDTH, HEIGHT = 400, 200


def convert_to_xy(theta, phi):
    x = (phi) / 2 / np.pi * WIDTH
    y = (theta / np.pi) * HEIGHT
    return x, y


def convert_to_spherical(x, y):
    phi = x * 2.0 / WIDTH * np.pi
    theta = y / HEIGHT * np.pi
    return theta, phi


@partial(jax.vmap, in_axes=(0, 0, None, None))
@partial(jax.vmap, in_axes=(None, None, 0, 0))
def compute_sh(n, m, theta, phi):
    complex_sh = jax.scipy.special.sph_harm_y(
        n, jnp.abs(m), theta, phi, n_max=_MAX_DEGREE
    )
    real_sh = jnp.where(
        m < 0,
        jnp.sqrt(2) * (-1) ** m * complex_sh.imag,
        jnp.where(
            m == 0,
            complex_sh.real,
            jnp.sqrt(2) * (-1) ** m * complex_sh.real,
        ),
    )
    return real_sh


@jax.jit
def render_points(nm, points, coefs):
    """Render the image from the parameters."""
    sh_values = compute_sh(
        nm[:, jnp.newaxis, 0],
        nm[:, jnp.newaxis, 1],
        points[:, jnp.newaxis, 0],
        points[:, jnp.newaxis, 1],
    )
    rgb = (
        jax.numpy.tanh(jnp.sum(sh_values * coefs[:, jnp.newaxis, :], axis=0)) + 1.0
    ) / 2.0
    return rgb


def generate_nm(max_degree):
    """Generate the (m, n) pairs for spherical harmonics up to max_degree."""
    n_m = []
    for n in range(max_degree + 1):
        for m in range(-n, n + 1):
            n_m.append((n, m))
    return jnp.array(n_m)  # shape (num_harmonics, 2)


def main():
    img = Image.open("earth.jpg")
    img_array = jnp.array(img)
    key = jax.random.PRNGKey(0)
    nm = generate_nm(_MAX_DEGREE)  # shape (num_harmonics, 2)
    coefs = jax.random.normal(key, (nm.shape[0], 3)) * 0.1
    points = fibonacci.fibonacci_sphere_spherical(_NUM_POINTS)  # shape (_NUM_POINTS, 2)

    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(coefs)

    @jax.jit
    def loss_fn(coefs):
        rendered = render_points(
            nm,
            points,
            coefs,
        )
        x, y = convert_to_xy(points[:, 0], points[:, 1])
        x = jnp.clip(x.astype(int), 0, WIDTH - 1)
        y = jnp.clip(y.astype(int), 0, HEIGHT - 1)
        gt_array = img_array[y, x, :] / 255.0
        return jnp.mean(optax.losses.l2_loss(rendered, gt_array))

    @jax.jit
    def update(params, opt_state):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    for i in range(_ITERATIONS):
        if i % 10 == 0:
            x = jnp.linspace(0, WIDTH - 1, WIDTH)
            y = jnp.linspace(0, HEIGHT - 1, HEIGHT)
            coords = jnp.array(jnp.meshgrid(y, x)).T.reshape(-1, 2)
            theta, phi = convert_to_spherical(coords[:, 1], coords[:, 0])
            points = jnp.column_stack((theta, phi))
            r_image = render_points(
                nm,
                points,
                coefs,
            )
            r_image = r_image.reshape((HEIGHT, WIDTH, 3)) * 255.0
            img = Image.fromarray(np.array(r_image.astype(jnp.uint8)))
            img.save(f"output_{i:04d}.png")

        coefs, opt_state = update(coefs, opt_state)
        current_loss = loss_fn(coefs)
        print(f"Step: {i}, Loss: {current_loss}")


if __name__ == "__main__":
    main()
