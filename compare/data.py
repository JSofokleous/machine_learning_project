import pandas as pd

## 1A: LOAD DATA
# Load the passenger data
df = pd.read_csv('compare/passengers.csv')

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

def load_clean_data():
    return df    