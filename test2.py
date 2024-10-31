import numpy as np
import cv2
import random
from Gene import Gene

# Load target image
target_image_path = "target_image_eren.jpg"
target_image = cv2.imread(target_image_path)
height, width, _ = target_image.shape

target_image = cv2.resize(target_image, (width // 4, height // 4))
height, width, _ = target_image.shape

# Parameters
generations = 10000  # Large number to evolve over time
mutation_rate = 0.1

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

class GDObject:
    def __init__(self):
        self.shape = random.choice(["circle", "triangle", "square"])
        self.x = random.randint(0, width - 1)
        self.y = random.randint(0, height - 1)
        self.size = random.randint(1, 30)
        self.color = target_image[self.y, self.x].tolist()
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.5, 1.0)

    def mutate(self):
        # Small random mutation
        if random.random() < mutation_rate:
            self.x = min(max(0, self.x + random.randint(-5, 5)), width - 1)
            self.y = min(max(0, self.y + random.randint(-5, 5)), height - 1)
            self.size = min(30, max(1, self.size + random.randint(-3, 3)))
            self.rotation = (self.rotation + random.randint(-10, 10)) % 360
            self.opacity = min(
                1.0, max(0.5, self.opacity + random.uniform(-0.05, 0.05))
            )
            self.color = target_image[self.y, self.x].tolist()

    def render(self, canvas):
        color = (
            int(self.color[0] * self.opacity),
            int(self.color[1] * self.opacity),
            int(self.color[2] * self.opacity),
        )

        if self.shape == "circle":
            cv2.circle(canvas, (self.x, self.y), self.size, color, -1)
        elif self.shape == "triangle":
            pts = np.array(
                [
                    [
                        self.x + self.size * np.cos(np.deg2rad(self.rotation)),
                        self.y + self.size * np.sin(np.deg2rad(self.rotation)),
                    ],
                    [
                        self.x + self.size * np.cos(np.deg2rad(self.rotation + 120)),
                        self.y + self.size * np.sin(np.deg2rad(self.rotation + 120)),
                    ],
                    [
                        self.x + self.size * np.cos(np.deg2rad(self.rotation + 240)),
                        self.y + self.size * np.sin(np.deg2rad(self.rotation + 240)),
                    ],
                ],
                np.int32,
            )
            cv2.fillPoly(canvas, [pts], color)
        elif self.shape == "square":
            top_left = (self.x - self.size // 2, self.y - self.size // 2)
            bottom_right = (self.x + self.size // 2, self.y + self.size // 2)
            cv2.rectangle(canvas, top_left, bottom_right, color, -1)


def fitness(objects):
    # Create a blank canvas
    canvas = np.zeros_like(target_image, np.uint8)

    # Render each object
    for obj in objects:
        obj.render(canvas)

    # Calculate the fitness as the color difference between the target and generated images
    diff = np.sum(np.abs(target_image - canvas))
    return diff


# Start with an empty list of GDObjects
objects = []
best_fitness = float("inf")  # Start with a very high fitness

# Evolutionary loop
for gen in range(generations):
    # Add a new object
    new_object = Gene(target_image, mutation_rate)
    objects.append(new_object)

    improved = False  # Track if we find an improvement

    # Try mutations on the new object
    for _ in range(100):  # Limit mutations to avoid infinite loops
        # Mutate the object and calculate fitness
        new_object.mutate()
        new_fitness = fitness(objects)

        # Check if fitness improved
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            improved = True
            break  # Keep this mutation if it improved fitness

    if not improved:
        # If no improvement, remove the latest object
        objects.pop()

    # Display progress
    if gen % 10 == 0:
        print(f"Generation {gen}, Fitness {best_fitness}")
        # Show the current best rendering
        # canvas = np.zeros_like(target_image, np.uint8)
        # Canvas should be initialized with the average color of the target image
        canvas = np.full_like(target_image, np.mean(target_image, axis=(0, 1), dtype=np.uint8))
        
        for obj in objects:
            obj.render(canvas)
        cv2.imshow("Generated Image", canvas)
        cv2.waitKey(1)

# Display the final result
canvas = np.zeros_like(target_image, np.uint8)
for obj in objects:
    obj.render(canvas)

cv2.imshow("Final Generated Image", canvas)
cv2.imwrite("output.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
