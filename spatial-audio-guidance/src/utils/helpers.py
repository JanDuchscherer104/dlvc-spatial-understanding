def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points in 3D space."""
    return ((point1[0] - point2[0]) ** 2 + 
            (point1[1] - point2[1]) ** 2 + 
            (point1[2] - point2[2]) ** 2) ** 0.5

def normalize_vector(vector):
    """Normalize a 3D vector."""
    magnitude = calculate_distance(vector, (0, 0, 0))
    if magnitude == 0:
        return (0, 0, 0)
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)

def convert_to_cartesian(polar_coordinates):
    """Convert polar coordinates (radius, theta, phi) to Cartesian coordinates (x, y, z)."""
    r, theta, phi = polar_coordinates
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return (x, y, z)

def log_message(message):
    """Log a message to the console."""
    print(f"[LOG] {message}")