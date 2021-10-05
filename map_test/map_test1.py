import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import plotly.express as px


country_path = r'/Users/sairana/Documents/Project0/Collision_Avoid/OSC_Test_Realtime/map_test/testcsv.csv'
df = pd.read_csv(country_path,sep = r'\s*,\s*', engine = 'python')

#print(df)


df_geo = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude,df.longitude))



print(df_geo)

world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

axis = world_data[world_data.continent == 'Africa'].plot(color = 'lightblue',edgecolor = 'black')

df_geo.plot(ax = axis, color = 'black')
plt.title('WA coiuntries')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9,6)
fig.savefig('matplot.png',dpi=200)

f = px.choropleth(df,locationmode='country names',locations=df['country'],scope='africa',color=df['country'])


plt.show()

f.show()