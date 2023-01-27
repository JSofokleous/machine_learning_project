import numpy as np

def get_sample(norm, norm_bool):
    # Example passenger features (not used later on)
    # Jack = np.array([0, 20, 0, 0, 0])
    # Rose = np.array([1, 17, 1, 0, 0])
    # example_passengers = np.array([Jack, Rose])
    # example_passengers = norm.transform(example_passengers)

    sample_age = 0
    while sample_age <= 0:
        sample_age = int(input("\nWhat is the age of your character? "))
    sample_sex = -1
    while sample_sex < 0 or sample_sex > 1:
        sample_sex = float(input("\nWhat is the sex of your character? (Please enter 0 if male, 1 if female, or any number in between if non-binary): "))
    sample_class = 0
    while sample_class != 1 and sample_class != 2 and sample_class != 3:
        sample_class = float(input("\nWhat is the class of your character? (Please enter 1 for first, 2 for second, or 3 for third): "))

    # Determine class from input
    sample_first_class = 0
    sample_second_class = 0
    if sample_class == 1: sample_first_class = 1
    elif sample_class == 2: sample_second_class = 1

    # Create and normalise 2D array for sample character featires
    sample_features = np.array([[sample_sex, sample_age, sample_first_class, sample_second_class, sample_first_class]])
    if norm_bool == True:
        print("~~~~NORMALISING~~~~")
        sample_features = norm.transform(sample_features)
    return sample_features