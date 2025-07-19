import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_name = "balancedface_results.csv"
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"The file '{file_name}' was not found. Please make sure the file is in the correct directory.")
    exit()


# Set modern plot styles
sns.set_theme(style="whitegrid")

# 1. How many identities per race
identities_per_race = df.groupby('race')['identity'].nunique().reset_index()
identities_per_race = identities_per_race.sort_values(by='identity', ascending=False)

plt.figure(figsize=(12, 7))
ax = sns.barplot(data=identities_per_race, x='race', y='identity', palette='viridis')
plt.title('Number of Unique Identities per Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Number of Identities', fontsize=12)
plt.xticks(rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fontsize=10)
plt.tight_layout()
plt.savefig('identities_per_race.png')
print("Identities per race:")
print(identities_per_race)
print("\n" + "="*50 + "\n")


# 2. Gender distribution per race
# Get the row with the highest confidence score for each identity
idx = df.groupby(['identity'])['conf_score'].transform(max) == df['conf_score']
df_highest_conf = df[idx].drop_duplicates(subset=['identity'])


gender_distribution_per_race = df_highest_conf.groupby(['race', 'gender']).size().reset_index(name='count')
gender_distribution_per_race = gender_distribution_per_race.sort_values(by=['race', 'count'], ascending=[True, False])

plt.figure(figsize=(14, 8))
ax = sns.barplot(data=gender_distribution_per_race, x='race', y='count', hue='gender', palette='plasma')
plt.title('Gender Distribution per Race (based on highest confidence score)', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fontsize=10)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('gender_distribution_per_race.png')
print("Gender distribution per race:")
print(gender_distribution_per_race)
print("\n" + "="*50 + "\n")


# 3. Total gender distribution
total_gender_distribution = df_highest_conf['gender'].value_counts().reset_index()
total_gender_distribution.columns = ['gender', 'count']

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=total_gender_distribution, x='gender', y='count', palette='cividis')
plt.title('Total Gender Distribution (based on highest confidence score)', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
for container in ax.containers:
    ax.bar_label(container, fontsize=10)
plt.tight_layout()
plt.savefig('total_gender_distribution.png')
print("Total gender distribution:")
print(total_gender_distribution)
print("\n" + "="*50 + "\n")


# 4. Statistics of number of images per race
images_per_identity = df.groupby(['race', 'identity']).size().reset_index(name='image_count')
image_stats_per_race = images_per_identity.groupby('race')['image_count'].describe()

plt.figure(figsize=(12, 7))
sns.boxplot(data=images_per_identity, x='race', y='image_count', palette='magma')
plt.title('Distribution of Number of Images per Identity for Each Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Number of Images per Identity', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images_per_identity_distribution.png')
print("Statistics of number of images per race:")
print(image_stats_per_race[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'}))
print("\n" + "="*50 + "\n")


# 5. Statistics of img_width per race
width_stats_per_race = df.groupby('race')['img_width'].describe()

plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='race', y='img_width', palette='inferno')
plt.title('Distribution of Image Width per Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Image Width', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('image_width_distribution.png')
print("Statistics of image width per race:")
print(width_stats_per_race[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'}))
print("\n" + "="*50 + "\n")


# 6. Statistics of img_height per race
height_stats_per_race = df.groupby('race')['img_height'].describe()

plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='race', y='img_height', palette='mako')
plt.title('Distribution of Image Height per Race', fontsize=16)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Image Height', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('image_height_distribution.png')
print("Statistics of image height per race:")
print(height_stats_per_race[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'}))
print("\n" + "="*50 + "\n")