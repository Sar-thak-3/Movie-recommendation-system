import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
columns_names = ['user_id', 'item_id','rating','timestamp']
df = pd.read_csv('u.data' , sep='\t' , names=columns_names)

df2= pd.read_csv('u.item',encoding = "ISO-8859-1" , sep="\|" , header=None)
# print(df2.head())
movies_titles = df2[[0,1]]  #here 0,1 are columns name
movies_titles.columns = ['item_id','title']
# print(movies_titles.head())

merged = pd.merge(df , movies_titles , on='item_id')

# Exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# it is the mean of each movies rated in sorted order
# print(merged.groupby('title').mean()['rating'].sort_values(ascending=False))

# now cleaning of data -> as we do not want those movies which are only rated by 1 or very few people and got 5 rating because it might be incorrect data!
# to check how much people watch 5 star 

ratings = pd.DataFrame(merged.groupby('title').mean()['rating'])
ratings['number_of_ratings'] = pd.DataFrame(merged.groupby('title').count()['rating'])
# as we get many movies withonly 1/2 reviews whivh can act as outliers

sns.jointplot(x='rating' , y='number_of_ratings' , data=ratings , alpha=0.5)
plt.show()
# jointplot.png so it shows higher rating less number of rating which have no significance


# ************************4th video****************

# Matrix -
#           -------------movie titles--------------
# User id's
#   |       _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
#   |            Rating by each user to each movies
#   |
#   |
#   |

moviemat = merged.pivot_table(index='user_id' , columns='title' , values='rating')
print(moviemat.shape)
print(moviemat.head())

# start wars(1977) user ratings
startwars_user_ratings = moviemat['Star Wars (1977)']

# correlation with startwars_user_ratings
similar_to_starwars = moviemat.corrwith(startwars_user_ratings)
# Contains Nan values becoz there are persons which not rated both the movies
corr_starwars = pd.DataFrame(similar_to_starwars , columns=['Correlation'])
corr_starwars.dropna(inplace=True) # just remove Nan values
print(corr_starwars)
# so we suggest those movies which have highes correlation with startwars movie

print(corr_starwars.sort_values('Correlation' , ascending=False).head(10))
# now as some movies have perfect correlation of 1 bcoz maybe they are rated by very few people
# so we need to filter those less number rated movies

# filter and consider only those movies which are rated by more than 100 people

print("corr_starwars.join(ratings['number_of_ratings'])")
corr_starwars_with_numberofratings = corr_starwars.join(ratings['number_of_ratings'])

# this produce movies which have number_of ratings more than 100
# this is what we need - > Movie recommendation system
print(corr_starwars_with_numberofratings['number_of_ratings']>100)
print(corr_starwars_with_numberofratings[corr_starwars_with_numberofratings['number_of_ratings']>100].sort_values('Correlation' , ascending=False))




# Predict function
def predictFunction(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_to_movie , columns=['Correlation'])
    corr_movie.dropna(inplace=True)

    corr_movie = corr_movie.join(ratings['number_of_ratings'])
    predictions = corr_movie[corr_movie['number_of_ratings']>100].sort_values('Correlation' , ascending=False)
    return predictions
