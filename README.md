# reality_market
проект по анализу рынка рынка компьютерных игр

Анализ продажи игр в интернет-магазине.
Цель:

проанализировать рынок игр на разных платформах (приставках) для определения более значимых продуктов, которые необходимо продвигать и определение факторов, которые непосредственно влияют на продажи.


import pandas as pd <br>
import seaborn as sns <br>
import matplotlib.pyplot as plt <br>
from scipy import stats as st <br>
from matplotlib.axes._axes import _log as matplotlib_axes_logger <br>


data = pd.read_csv('/datasets/games.csv')


data.head(5)


data.info()

Приведем названия столбцов к нижнему регистру

data.columns = data.columns.str.lower()
data.head(5)

data.columns

при первичном анализе данных видно, что в некоторых стоблцах данные привелены в разных типах данных. Столбец year of release нужно привести к целым числам

data['year_of_release'] = data['year_of_release'].astype('Int64')

данные в стоблцe user_score приведены с типом данных object, нужно привести к типу вещественных чисел. Но проверим сначала уникальные данные в столбцe чтобы избежать ошибок при изменении типа данных

data['user_score'].unique()

в стоблце обнаружены странные данные tbd, в интернете можно найти что это сокращение от to be determined, по всей видимости таким образом не заполнили данные намеренно, когда рейтинг игрока еще не был опеределен. Вероятно, tbd можно оставить в таблице, но для удобства анализа заменить на любое вещественное число, например на 0000. Такая замена позволит далее эту замену легко отфильтровать при необходимости.

data['user_score']=pd.to_numeric(data['user_score'], errors='coerce')
data['user_score'].unique()

data['user_score'] = data['user_score'].astype(float)


 
обработал tbd с помощью to_numeric

data['user_score'].unique()

проверим пропуски в данных

data.isna().sum()

проверим дубликаты в таблице

data.duplicated().sum()

есть пропуски в данных в стоблце year of release, обработаем эти пропуски

data[data['year_of_release'].isna()]

из таблицы видно, что игры в разные годы выходили на разных приставках, можем замегить попуски в данных на год выпуска той же игры на другой приставке

for i in data[data['year_of_release'].isnull() == True].index:  
    data['year_of_release'][i] = data.loc[data['name'] == data['name'][i], 'year_of_release'].max()

приверим пропуски после такой замены

data['year_of_release'].isna().sum()

пропуски остались, но заменить их пока нечем, оставим эти пропуски. Посмотрим на пропуски в других столбцах

data[data['critic_score'].isna()]

есть пропуски в данных в critic_score, user_score и rating. Рейтинг игроков могли просто не считать в некоторых случаях или сценарием игры он не предполагался. Заменить эти пропуски нечем, веронятнее всего эти попуски не окажут существенного влиятияни на анализ, их можно оставить.

в этих стоблцах всего по два пропуска, это походе на техническую ошибку и их можно игнорировать, на анализе данных не скажется

Посчитаем суммарные продажи во всех регионах и запишем их в отдельный столбец. Для этого создадим новый столбец total_sales

data['total_sales'] = data['na_sales'] + data['eu_sales'] + data['jp_sales'] + data['other_sales']

Для вычисления суммарных продаж можно использовать метод sum с аргументом axis.
    
    
(data.head(10))

Посмотрим, сколько игр выпускалось в разные годы и важны ли данные за все периоды.

Используем метод pivot_table и построим график

games_on_period = data.pivot_table(index='year_of_release', values='name', aggfunc='count')
plt.figure(figsize=(12,6))
sns.lineplot(data=games_on_period)
plt.title("Количество игр в разные годы")
plt.xlabel("Год выпуска")
plt.ylabel("Количество игр")
plt.legend('');

из нашего графика видно, что количество игр начало расти после 1995 года и росло до 2010 года, можно предположить что этот рост связан с быстрым развитием рынка домашних компьюетров и приставок. Снижение роста количества игр после 2010 года, скорее всего связано с развитием рынка мобильных телефонов.

Посмотрим, как менялись продажи по платформам. Выберем платформы с наибольшими суммарными продажами и построим распределение по годам. Определим, за какой характерный срок появляются новые и исчезают старые платформы.

посмотрим продажи по платформам

(data.pivot_table(index='platform', values='total_sales', aggfunc='sum').reset_index().plot(kind='bar', x='platform', y='total_sales'))
plt.ylabel('total_sales')


посмотрим распределение продаж по годам


для начала построим таблицу по продажам на разных платформах и сделаем сортировку по самым результативным платформам

best_platforms = data.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False).head(10)
best_platforms = best_platforms.reset_index().rename_axis(None, axis=1)
print (best_platforms)

построим таблицу по продажам по платформам с 1995 года когда количество игр начало расти

def year_sales_platforms(name, data):
    slicee = data[(data['platform'] == name) & (data['year_of_release'] > 1995)]
    total = slicee.pivot_table(index='year_of_release', values='total_sales', aggfunc='sum').sort_values('year_of_release', ascending=False)
    return total

построим график по продажам игр на разных платформах с 1995 года

plt.figure(figsize=(12,6))
plt.title('Продажи платформ')
plt.xlabel('Годы выпуска')
plt.ylabel('Продажи')

for i in list(best_platforms['platform']):
    sns.lineplot(data=year_sales_platforms(i,data)['total_sales'], label=i)
    plt.legend()

Самые продаваемая платформа это PS2. Новые платформы появляются примерно за 5-7 лет, но с 2002 года новые платформы появляются каждые год-два и исчезают за 10 лет.


top_platforms=data.groupby('platform').agg({'total_sales':'sum'}).sort_values(by='total_sales').tail(6).index.to_list() 
top_platforms

print (top_platforms)

fig, boxplot = plt.subplots(figsize = (20,7))
boxplot=sns.boxplot(x=data.query('platform == @top_platforms')['platform'], y=data['year_of_release'])
boxplot.axes.set_title('Распределение популярных платформ по годам', size=20)
boxplot.set_xlabel('платформа', fontsize =17)
boxplot.set_ylabel('год_выпуска', fontsize=17);

<br/>
Построим график «ящик с усами» по глобальным продажам игр в разбивке по платформам

Для этого создадим новую переменную для сохранения самых популярных платформ и по ней

data_new = data.loc[(data['year_of_release'] >=2012)]
data_new.head(5)

list_of_top5 = ['PS2', 'DS', 'Wii', 'PS3', 'X360']
data_top_5_platforms = data[data['platform'].isin(['PS2', 'DS', 'Wii', 'PS3', 'X360'])]
data_top_5_platforms = data_top_5_platforms[data_top_5_platforms['total_sales']<1.4]

впостроил график для оценки динами продаж по платформам в актуальном периоде
</div>

data_new.pivot_table(index='platform', values='name', aggfunc='count').reset_index().plot(kind='bar', x='platform', y='name')
plt.ylabel('number of games')
plt.show()


sns.lineplot(hue = 'platform', y = 'total_sales', x = 'year_of_release',  data=data_new, estimator = 'sum', ci=None);

    
выполнил выбор периода ранее построения боксплотов
</div>

sns.boxplot(data=data_new, x='platform', y='total_sales')
plt.ylim(0, 3)
plt.show()

   
построил график по продажам в разбике по платформам с учетом актуального периода
</div>

data_top_5_platforms['total_sales'].describe()

plt.figure(figsize=(12,6))
sns.boxplot(data=data_top_5_platforms, x='platform', y='total_sales')
plt.title('Ящик с усами', fontsize=15)
plt.xlabel('Платформа', fontsize=12)
plt.ylabel('Глобальные продажи',fontsize=12)

Из графика видно, что больше всего продаж по платформе PS3, на втором месте Х360.

Посмотрим, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Построим диаграмму рассеяния и посчитаем корреляцию между отзывами и продажами. 

print(data['platform'].unique())

выберем данные по продажам за акутальный период

data_new = data.loc[(data['year_of_release'] >=2012)]
print (data_new.head(5))

platforms = ['PS4', 'PS3', 'X360', 'XOne', 'WiiU', 'Wii']
for i in platforms:
    x_critic = data_new.loc[(data_new['platform']==i) & (data_new['critic_score'])]['critic_score']
    y_critic = data_new.loc[(data_new['platform']==i) & (data_new['critic_score'])]['total_sales']
    x_user = data_new.loc[(data_new['platform']==i) & (data_new['user_score'])]['user_score']
    y_user = data_new.loc[(data_new['platform']==i) & (data_new['user_score'])]['total_sales']
    
    fig = plt.figure(figsize=(13.5, 4.5))
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(x_critic, y_critic)
    ax1.set_title('Зависимость продаж от оценок критиков\n на платформе '+i, fontsize=14, fontweight="bold")
    print('корреляция Пирсона от оценок критиков\n'+i,'-', x_critic.corr(y_critic))

    ax2 = fig.add_subplot(122)
    ax2.scatter(x_user, y_user)
    ax2.set_title('Зависимость продаж от оценок пользователей\n на платформе '+i, fontsize=14, fontweight="bold")
    print('корреляция Пирсона от оценок пользователей\n'+i, '-', x_user.corr(y_user))

У платформы WII данные не показательны, так как мало данных.
У платформ с наилучшими продажами слабая зависимость от рейтинга критиков. 
У этих же платформ очень слабая зависимость от рейтинга пользователей.
Можно сделать вывод, что от мнения критиков зависит успех продаж игр. Мнения критиков можно использовать для продвижения.


Посмотрим на общее распределение игр по жанрам. Что можно сказать о самых прибыльных жанрах? Выделяются ли жанры с высокими и низкими продажами? Для этого построим таблицу по жанрам и продажам и сделаем сортировку по убыванию.

    
применил анализ на основе акутального периода
</div>

games_genre = data_new.pivot_table(index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)
games_genre = games_genre.reset_index().rename_axis(None, axis=1)
print (games_genre)

Визуализиуем на графике продажи по жанрам.

plt.figure(figsize=(12,6))
plt.title('Продажи по жанрам ',fontsize=15)
sns.barplot(data=games_genre, x='genre', y='total_sales')
plt.xlabel('Жанры',fontsize=12)
plt.ylabel('Продажи',fontsize=12)

Лучшие продажи у игр в жанре "Action", жанр "Strategy" демонстрирует худшие продажи.

рассмотрел медианные данные по жанрам
</div>

display (data_new.groupby('genre').median()['total_sales'].sort_values(ascending=False))
data.boxplot(by='genre', column='total_sales', figsize=(10,7))
plt.ylim(0,3)

по медианным продажам по жанрам видно,что для некоторых жанров медианные продажи почти одинаковые
1.добавил ylim в график.
    
2.сделал анализ по жанрам на основе актуального периода.
Определим для пользователя каждого региона самые популярные платформы (топ-5). Самые популярные жанры (топ-5).
Выясним, влияет ли рейтинг ESRB на продажи в отдельном регионе.
Для начала определим продажи по платформам в каждом регионе.


regions = ['na_sales', 'eu_sales', 'jp_sales']
name = ['США', 'ЕС', 'Япония']

platform_top_5 = []

for i in regions:
    data_platform = data_new.groupby('platform')[i].sum().sort_values(ascending = False)
    data_platform_1 = data_platform.head()
    data_platform_1.loc['Other'] =  data_platform.sum() - data_platform_1.sum()
    data_platform_2 = 100*data_platform_1/data_new[i].sum()
    platform_top_5.append(data_platform_2)
    new_data = pd.DataFrame(platform_top_5).round(2).transpose()


new_data.plot(kind='pie', autopct=lambda p: '{:.1f}%'.format(round(p)) if p > 0 else'', subplots=True, startangle=90, legend = False, fontsize=14, figsize =(15,15), shadow=True, title=name)
plt.show()

new_data


    
добавил figsize в графики
</div>
В США лидер по продажам платформа Х360, в ЕС лидер платформа PS4, в Японии лучшие продажи у 3DS, при этом эта платформа показывает худшие результаты по продажам в США и ЕС.

Теперь рассмотрим продажи в регионах по жанрам

regions = ['na_sales', 'eu_sales', 'jp_sales']
name = ['США', 'ЕС', 'Япония']

genre_top_5 = []


for i in regions:
    data_genre = data_new.groupby('genre')[i].sum().sort_values(ascending = False)
    data_genre_1 = data_genre.head()
    data_genre_1.loc['Other'] =  data_genre.sum() - data_genre_1.sum()
    data_genre_2 = 100*data_genre_1/data_new[i].sum()
    genre_top_5.append(data_genre_2)
    new_data_genre = pd.DataFrame(genre_top_5).round(2).transpose()


new_data_genre.plot(kind='pie', autopct=lambda p: '{:.1f}%'.format(round(p)) if p > 0 else'', subplots=True, startangle=90, legend = False, fontsize=14, shadow=True, title=name)
plt.show()

new_data_genre


В США и ЕС самые популярные жанры Action и Shooter. В Японии самый популярный жанр Role-Playing. 


Выясним, влияет ли рейтинг ESRB на продажи в отдельном регионе.

data_new['rating'].head(10)

data_new['rating'].unique()

data_new['rating'].isna()



data_new['rating'] = data_new['rating'].fillna(0000)

data_new['rating'].unique()

na_rating = data_new.groupby('rating').sum().na_sales
eu_rating = data_new.groupby('rating').sum().eu_sales
jap_rating = data_new.groupby('rating').sum().jp_sales
na_rating.head(10)


eu_rating.head(10)


jap_rating.head(10)

1.изменил анализ на акутальный период
    
2.предположу, что ESRB это американская негосударственная организация, определяющая ретинги для игр, продающихся в США И Канаде. Эти рейтинги могут не иметь значения для покупателей в Японии.

esrb_sales = data_new.pivot_table(index='rating', values=['na_sales', 'eu_sales', 'jp_sales'], aggfunc='sum')
esrb_sales.sort_values(by='eu_sales')

Видно, что рейтинг ESRB влияет на продажи в США и в ЕС. В Японии влияние этого рейтинга на продажи не значительное.

Заменил пропуски в ретинге на значение заглушку. Выяснил что игры без рейтинга более покупают в США. 
</div>

Проверим гипотезы
•	Средние пользовательские рейтинги платформ Xbox One и PC одинаковые;
•	Средние пользовательские рейтинги жанров Action (англ. «действие», экшен-игры) и Sports (англ. «спортивные соревнования») разные.


Нулевая гипотеза:
Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
Альтернативная гипотеза:
Средние пользовательские рейтинги платформ Xbox One и PC различаются.

Для проерки гипотез нам потребуются новые переменные и данные подберем за последние 10 лет и вычислим средний рейтинг игрока ХOne.

print(data['platform'].unique())

xone = data[(data['platform']=='XOne') & (data['year_of_release']>2012)]['user_score']
pc = data[(data['platform']=='PC') & (data['year_of_release']>2012)]['user_score']
print (xone.mean())

print (pc.mean())

проверим гипотезы с помощью метода ttest_ind, зададим значение alpha .01

alpha = .05

results = st.ttest_ind(xone.dropna(), pc.dropna(), equal_var=False)

print('p-значение:', results.pvalue)


if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")

Видно, что p-значение менее 4%, то есть с вероятностью в 4% можно получить различность рейтингов двух платформ.

Нулевая гипотеза: 
Средние пользовательские рейтинги жанров Action и Sports одинаковые
Альтернативная гипотеза:
Средние пользовательские рейтинги жанров Action и Sports различаются

print (data['genre'].unique())

genre_action = data[(data['genre']=='Action') & (data['year_of_release']>2012)]['user_score']
genre_sports = data[(data['genre']=='Sports') & (data['year_of_release']>2012)]['user_score']

print (genre_action.mean())

print (genre_sports.mean())

alpha = .05

results = st.ttest_ind(genre_action.dropna(), genre_sports.dropna(), equal_var=False)

print('p-значение:', results.pvalue)


if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


Видно, что при p-value более 3% практически нет вероятности получить одинаковые средние пользовательскнеи рейнтинги по этим жанрам.

Общий вывод.
Жанр и отзывы критиков являются важной частью маркетинговой стратегии для успешных продаж.

Платформы живут на рынке около 10 лет. Налюдается падение продаж на платформах после 2010 года в связи с развитием мобильных устройств.

Лучше всего игры продавались на PS2, X360 и PS3.

Pейтинг ESRB влияет на продажи в США и в ЕС. В Японии влияние этого рейтинга на продажи не значительное.

По портрету пользователя в США самые продаваемые игры в жанрах Action и Shooter. В Японии самые продаваемые игры в жанре Role-Playing. 

В США лидер по продажам платформа Х360, в ЕС лидер платформа PS4, в Японии лучшие продажи у 3DS, при этом эта платформа показывает худшие результаты по продажам в США и ЕС.


