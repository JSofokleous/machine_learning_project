import pandas as pd

def load_house():
    streeteasy = pd.read_csv('data/rent.csv')
    df = pd.DataFrame(streeteasy)

    # print(labels.rent.mean())
    # print(features.bathrooms.unique())
    # print(features.isnull().values.any())

    df['max_rent'] = df.rent.apply(lambda x: 1 if x >= 5000 else 0)
    df['one_bed'] = df.bedrooms.apply(lambda x: 1 if x == 1 else 0)
    df['two_or_more_bed'] = df.bedrooms.apply(lambda x: 1 if x >= 2 else 0)
    df['three_or_more_bed'] = df.bedrooms.apply(lambda x: 1 if x >= 3 else 0)
    df['sub10mins_to_subway'] = df.min_to_subway.apply(lambda x: 1 if x <= 10 else 0)

    # df.drop(columns=['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs'], inplace=True)

    labels = df['max_rent']
    features = df[[ 'has_roofdeck', 'has_doorman', 'has_patio', 'has_gym', 'one_bed', 'two_or_more_bed', 'three_or_more_bed', 'sub10mins_to_subway' ]]
    # features = df[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'has_roofdeck', 'has_doorman', 'has_patio', 'has_gym', 'one_bed', 'two_or_more_bed', 'three_or_more_bed', 'sub10mins_to_subway' ]]

    # print(features.head())
    # print(labels.head())
    # print(df.tail())

    return features, labels 


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
    features = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']]
    labels = df['Survived']
    
    return features, labels

    # df.to_csv('compare/out.csv', index=False)
    
