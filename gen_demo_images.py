from chemvision.data.synthetic_generator import XRDImageGenerator

gen = XRDImageGenerator(seed=0)
samples = gen.generate_temperature_series([200, 400, 600], output_dir="demo_images/")
for s in samples:
    print(s.image_path)
