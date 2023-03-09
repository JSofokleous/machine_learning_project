import pandas as pd

def list_feature_names(features, binary):
    feature_names = {}
    for i in range(len(features.columns)):
        feature_names[features.columns[i]] = binary[i]
    print("Feature names: ", feature_names)
    feature_names_list = [name for name in feature_names]

    return feature_names, feature_names_list


def load_house(budget):
    # Load data into dataframe
    streeteasy = pd.read_csv('data/rent.csv')
    df = pd.DataFrame(streeteasy)

    # Features
    df['max_rent'] = df.rent.apply(lambda x: 0 if x >= budget else 1)
    labels = df['max_rent']

    # Labels
    df['one_bed'] = df.bedrooms.apply(lambda x: 1 if x == 1 else 0)
    df['two_or_more_bed'] = df.bedrooms.apply(lambda x: 1 if x >= 2 else 0)
    features = df[['size_sqft', 'min_to_subway', 'one_bed', 'two_or_more_bed', 'has_patio', 'has_gym']]
    binary = [0, 0, 1, 1, 1, 1]

    print('\nAVG SIZE: ', df.size_sqft.mean())
    print('AVG TIME TO SUBWAY', df.min_to_subway.mean())

    return features, binary, labels 


def load_titanic():
    ## 1A: LOAD DATA
    # Load the passenger data
    df = pd.read_csv('data/passengers.csv')

    ## 1B: CLEAN DATA
    # Update sex column to numerical
    df.Sex.replace('male', 0, inplace=True)
    df.Sex.replace('female', 1, inplace=True)

    # Fill the nan values in the age column to be mean age
    df.Age.fillna(df.Age.mean(), inplace=True)

    # Create a first/second class column
    df['FirstClass'] = df.Pclass.apply(lambda x: 1 if x == 1 else 0)
    df['SecondClass'] = df.Pclass.apply(lambda x: 1 if x == 2 else 0)

    # Create a surname column (not applicable to LR model)
    df['Surname'] = df.Name.apply(lambda x: x.split()[0].strip(','))

    # Create a master column
    df['Master'] = df.Name.apply(lambda x: 1 if x.split()[1].strip('.') == 'Master' else 0)
        # Increases accuracy from 87.8% to 88.5% 
        # Increases F1 score from 81.6% to 83.1%

    df.drop(columns=['Pclass', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], inplace=True)
    
    # Sort data into desired labels and features: ['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']
    features = df[['Sex', 'Age', 'FirstClass']]
    binary = [1, 0, 1]
    labels = df['Survived']
    
    return features, binary, labels
    
