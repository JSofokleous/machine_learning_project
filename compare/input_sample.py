def binary_prompt(dict, feature_names):
    for name in feature_names:
        while dict[name] != 0 and dict[name] != 1:
            dict[name] = int(input("\n{}: ".format(name)))


def get_binary_sample(feature_names):
    # Dictionary of all features and their values initally set to -1
    dict = {}
    for feature in feature_names:
        dict[feature] = -1

    binary_prompt(dict, feature_names)

    # Return a list of features, which where selected 
    list = []
    for i in dict:
        if i in feature_names:
            list.append(dict[i])
    return list


  # Promt user for sample features if applicable
    # if 'Age' in feature_names:
    #     while dict['Age'] <= 0:
    #         dict['Age'] = int(input("\nWhat is the age of your character? "))

    # if 'Sex' in feature_names:
    #     while dict['Sex'] < 0 or dict['Sex'] > 1:
    #         dict['Sex'] = float(input("\nWhat is the sex of your character? (Please enter 0 if male, 1 if female, or any number in between if non-binary): "))
    
    # if ('FirstClass' in feature_names) or ('SecondClass' in feature_names):
    #     sample_class = 0
    #     while sample_class != 1 and sample_class != 2 and sample_class != 3:
    #         sample_class = float(input("\nWhat is the class of your character? (Please enter 1 for first, 2 for second, or 3 for third): "))
    #     if sample_class == 1: dict['FirstClass'] = 1
    #     elif sample_class == 2: dict['SecondClass'] = 1

    # if 'Master' in feature_names:
    #     while dict['Master'] < 0 or dict['Master'] > 1:
    #         dict['Master'] = float(input("\nDoes your character have the title master? (Please enter 1 if yes, 0 if no): ")) 