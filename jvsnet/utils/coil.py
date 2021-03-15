from . import complex_multiply


def combine_all_coils(image, sensitivity, coil_dim=0):
    """ Return Sensitivity combined images from all coils """
    combined = complex_multiply(
        sensitivity[..., 0], -sensitivity[..., 1], image[..., 0], image[..., 1]
    )
    return combined.sum(dim=coil_dim)


def project_all_coils(x, sensitivity, coil_dim=1):
    """ Return combined image to coil images """
    x = complex_multiply(
        x[..., 0].unsqueeze(coil_dim),
        x[..., 1].unsqueeze(coil_dim),
        sensitivity[..., 0],
        sensitivity[..., 1],
    )
    return x
