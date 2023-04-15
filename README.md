# Ex-No.5  FEATURE GENERATION
# AIM:
To read the given data and perform Feature Generation process and save the data to a file.
# EXPLANATION:
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target
# ALGORITHM:
## STEP 1:
Read the given Data.
## STEP 2:
Clean the Data Set using Data Cleaning Process.
## STEP 3:
Apply Feature Generation techniques to all the feature of the data set.
## STEP 4:
Save the data to the file.
# CODE:
## Data set:
## Ordinal Encoder:
import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("data.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

classes = ['Cold','Warm','Hot','Very Hot']

enc = OrdinalEncoder(categories = [classes])

enc.fit_transform(df[["Ord_1"]])

df['ord_1']=enc.fit_transform(df[["Ord_1"]])

df

## Label Encoder:
import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("data.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

classes = [0,1]

enc = OrdinalEncoder(categories = [classes])

enc.fit_transform(df[["Target"]])

df['target']=enc.fit_transform(df[["Target"]])

df

## Binary Encoder:
import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("data.csv")

!pip install category_encoders

from category_encoders import BinaryEncoder

be=BinaryEncoder()

newdata=be.fit_transform(df['bin_1'])

df1=pd.concat([df,newdata],axis=1)

df1

## OneHotEncoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("data.csv")

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

df1 = df.copy()

enc = pd.DataFrame(ohe.fit_transform(df1[['City']]))

df1 = pd.concat([df1,enc],axis=1)

df1

## Encoding Data Set:

## Ordinal Encoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Encoding Data.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

classes = ['Red','Blue','Green']

enc = OrdinalEncoder(categories = [classes])

enc.fit_transform(df[["nom_0"]])

df['Nom_0']=enc.fit_transform(df[["nom_0"]])

df

## Binary Encoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Encoding Data.csv")

!pip install category_encoders

from category_encoders import BinaryEncoder

be=BinaryEncoder()

newdata=be.fit_transform(df['bin_1'])

df1=pd.concat([df,newdata],axis=1)

df1

## Titanic Data Set:

## Ordinal Encoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

classes = [1,2,3]

enc = OrdinalEncoder(categories = [classes])

enc.fit_transform(df[["Pclass"]])

df['ord_2']=enc.fit_transform(df[["Pclass"]])

df

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

classes = ['C','Q','S',np.nan]

enc = OrdinalEncoder(categories = [classes])

enc.fit_transform(df[["Embarked"]])

df['ord']=enc.fit_transform(df[["Embarked"]])

df

# Label Encoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

le = LabelEncoder()

df['Name1']=le.fit_transform(df['Name'])

df

## Binary Encoder:

import pandas as pd

import numpy as np

import seaborn as sns

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

!pip install category_encoders

from category_encoders import BinaryEncoder

be=BinaryEncoder()

newdata=be.fit_transform(df['Sex'])

df1=pd.concat([df,newdata],axis=1)

df1

## OUTPUT: 
## Data set:
## Ordinal Encoder:
![Screenshot (43)](https://user-images.githubusercontent.com/127846109/232105205-13470955-e2f3-498b-b0a3-69e5dcb6c08e.png)
## Label Encoder:
![Screenshot (44)](https://user-images.githubusercontent.com/127846109/232105360-ba7fae4f-f02f-4a70-802a-f8a9b6a5ded3.png)
## Binary Encoder:
![Screenshot (45)](https://user-images.githubusercontent.com/127846109/232105455-d641886e-df96-4667-b7a5-ee9c98f1594d.png)
## OneHotEncoder:
![Screenshot (46)](https://user-images.githubusercontent.com/127846109/232111010-bed79ca0-90d3-4ba3-877a-c70e562cc591.png)
## Encoding data set:
## Ordinal Encoder:
![Screenshot (47)](https://user-images.githubusercontent.com/127846109/232106394-5990867e-d9aa-49dc-ac5e-cbfcaa12a48c.png)
## Binary Encoder:
![Screenshot (48)](https://user-images.githubusercontent.com/127846109/232106553-bdacbd76-1472-4472-99e1-e6417149925f.png)
## Titanic Data set:
## Ordinal Encoder:
![Screenshot (49)](https://user-images.githubusercontent.com/127846109/232109590-d474adb8-5c01-42e7-9b81-729caf73f913.png)
![Screenshot (50)](https://user-images.githubusercontent.com/127846109/232111083-ad9eda5f-7e63-4fd7-aab4-54091e8a70fb.png)
## Label Encoder:
![Screenshot (51)](https://user-images.githubusercontent.com/127846109/232111147-dce33617-fb99-43dc-b431-11786986942f.png)
## Binary Encoder:
![Screenshot (52)](https://user-images.githubusercontent.com/127846109/232111184-7c25fb84-c27e-4dd2-be9a-5676b3220c6d.png)
## RESULT:
Thus the Feature Generation for the given data set is executed and
output was verified successfully
