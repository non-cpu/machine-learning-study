import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('ml_for_finance/train_1.csv').fillna(0)

# print(train.head())

def parse_page(page):
	x = page.split('_')
	return ' '.join(x[:-3]), x[-3], x[-2], x[-1]

# print(parse_page(train.Page[0]))

l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
df.columns = ['Subject','Sub_Page','Access','Agent']

# print(df.head())

train = pd.concat([train, df], axis=1)
del train['Page']

# fig, ax = plt.subplots(figsize=(10, 7))
# # train.Sub_Page.value_counts().plot(kind='bar')
# # train.Access.value_counts().plot(kind='bar')
# train.Agent.value_counts().plot(kind='bar')
# plt.show()


idx = 39457

data = train.iloc[idx,0:-4]
name = train.iloc[idx,-4]
days = [r for r in range(data.shape[0])]


# window = 10

# fig, ax = plt.subplots(figsize=(10, 7))

# plt.ylabel('Views per Page')
# plt.xlabel('Day')
# plt.title(name)

# ax.set_yscale('log')
# ax.plot(days, data.values, color='grey')
# ax.plot(np.convolve(data, np.ones((window))/window, mode='valid'), color='black')

# plt.show()


# fig, ax = plt.subplots(figsize=(10, 7))

# plt.ylabel('Views per Page')
# plt.xlabel('Day')
# plt.title('Twenty One Pilots Popularity')

# ax.set_yscale('log')

# for country in ['de', 'en', 'es', 'fr', 'ru']:
#     idx = np.where((train['Subject'] == 'Twenty One Pilots')
#         & (train['Sub_Page'] == '{}.wikipedia.org'.format(country))
#         & (train['Access'] == 'all-access') & (train['Agent'] == 'all-agents'))
#     idx=idx[0][0]

#     data = train.iloc[idx,0:-4]
#     ax.plot(days, data.values, label=country)

# ax.legend()

# plt.show()


# from scipy.fftpack import fft

# data = train.iloc[:,0:-4]
# fft_complex = fft(data)

# # print(fft_complex.shape)

# fft_mag = [np.sqrt(np.real(x) * np.real(x) + np.imag(x) * np.imag(x)) for x in fft_complex]

# arr = np.array(fft_mag)

# fft_mean = np.mean(arr, axis=0)

# # print(fft_mean.shape)

# fft_xvals = [day / fft_mean.shape[0] for day in range(fft_mean.shape[0])]

# npts = len(fft_xvals) // 2 + 1
# fft_mean = fft_mean[:npts]
# fft_xvals = fft_xvals[:npts]

# fig, ax = plt.subplots(figsize=(10, 7))

# ax.plot(fft_xvals[1:], fft_mean[1:])

# plt.axvline(x=1./7, color='red', alpha=0.3)
# plt.axvline(x=2./7, color='red', alpha=0.3)
# plt.axvline(x=3./7, color='red', alpha=0.3)

# plt.show()


from pandas.plotting import autocorrelation_plot

data = train.iloc[:,0:-4]


# autocorrelation_plot(data.iloc[110])
# plt.title(' '.join(train.loc[110,['Subject', 'Sub_Page']]))

# plt.show()


# a = np.random.choice(data.shape[0], 1000)

# for i in a:
#     autocorrelation_plot(data.iloc[i])

# plt.title('1K Autocorrelations')

# plt.show()


from sklearn.model_selection import train_test_split

X = data.iloc[:,:500]
y = data.iloc[:,500:]

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.1, random_state=42)


# def mape(y_true, y_pred):
#     eps = 1
#     err = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
#     return err

# lookback = 50
# lb_data = X_train[:,-lookback:]
# med = np.median(lb_data, axis=1, keepdims=True)
# eps = 1
# print(mape(y_train, med))

# idx = 15000

# fig, ax = plt.subplots(figsize=(10, 7))

# ax.plot(np.arange(500), X_train[idx], label='X')
# ax.plot(np.arange(500, 550), y_train[idx], label='True')

# ax.plot(np.arange(500, 550), np.repeat(med[idx], 50), label='Forecast')

# plt.title(' '.join(train.loc[idx, ['Subject', 'Sub_Page']]))
# ax.set_yscale('log')
# ax.legend()

# plt.show()


# from statsmodels.tsa.arima.model import ARIMA

# model = ARIMA(X_train[0], order=(5,1,5))

# model = model.fit()

# # print(model.summary())

# idx = 0

# residuals = pd.DataFrame(model.resid)


# fig, ax = plt.subplots(figsize=(10, 7))

# ax.plot(residuals)

# plt.title('ARIMA residuals for 2NE1 pageviews')

# plt.show()


# residuals.plot(kind='kde', figsize=(10,7), title='ARIMA residual distribution 2NE1 ARIMA', legend=False)

# plt.show()


# predictions = model.forecast(50)

# fig, ax = plt.subplots(figsize=(10, 7))

# ax.plot(np.arange(480, 500), X_train[0,480:], label='X')
# ax.plot(np.arange(500, 550), y_train[0], label='True')

# ax.plot(np.arange(500, 550), predictions, label='Forecast')

# plt.title('2NE1 ARIMA forecasts')
# ax.set_yscale('log')
# ax.legend()

# plt.show()


# import simdkalman

# smoothing_factor = 5.0

# n_seasons = 7

# # state transition matrix A
# state_transition = np.zeros((n_seasons + 1, n_seasons + 1))
# # hidden level
# state_transition[0,0] = 1
# # season cycle
# state_transition[1,1:-1] = [-1.0] * (n_seasons - 1)
# state_transition[2:,1:-1] = np.eye(n_seasons - 1)

# print(state_transition)

# observation_model = [[1,1] + [0] * (n_seasons - 1)]

# print(observation_model)

# level_noise = 0.2 / smoothing_factor
# observation_noise = 0.2
# season_noise = 1e-3

# process_noise_cov = np.diag([level_noise, season_noise] + [0] * (n_seasons - 1)) ** 2
# observation_noise_cov = observation_noise ** 2

# print(process_noise_cov)

# print(observation_noise_cov)

# kf = simdkalman.KalmanFilter(
# 	state_transition = state_transition,
# 	process_noise = process_noise_cov,
# 	observation_model = observation_model,
# 	observation_noise = observation_noise_cov
# )

# result = kf.compute(X_train[0], 50)

# fig, ax = plt.subplots(figsize=(10, 7))
# ax.plot(np.arange(480, 500), X_train[0,480:], label='X')
# ax.plot(np.arange(500, 550), y_train[0], label='True')

# ax.plot(np.arange(500, 550), result.predicted.observations.mean, label='Predicted observations')
# ax.plot(np.arange(500, 550), result.predicted.states.mean[:,0], label='redicted states')
# ax.plot(np.arange(480, 500), result.smoothed.observations.mean[480:], label='Expected Observations')
# ax.plot(np.arange(480, 500), result.smoothed.states.mean[480:,0], label='States')

# ax.set_yscale('log')
# ax.legend()

# plt.show()
