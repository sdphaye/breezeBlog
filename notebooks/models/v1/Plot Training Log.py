# %%
%matplotlib inline

# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# %%
df = pd.read_csv('training_log.csv')
df

# %%
figure, ax = plt.subplots(figsize=(16,8))
df.rename(columns={
    'loss': 'Training Loss', 'val_loss': 'Validation Loss'
}).plot('epoch', ['Training Loss', 'Validation Loss'], ax=ax)
ax.set_title('Training Loss and Validation Loss over Time (Lower is Better)')
ax.set_ylabel('Loss Value')
ax.set_xlabel('Epoch')

# %%
