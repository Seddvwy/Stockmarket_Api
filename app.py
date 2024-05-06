from flask import Flask, jsonify, request
import pandas_ta as ta
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import asyncio
import aiohttp
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

saudia = [
    "4310.SR", "2222.SR", "2380.SR", "2381.SR", "4030.SR", "4200.SR", "1201.SR", "1202.SR", "1210.SR", "1211.SR", 
    "1301.SR", "1304.SR", "1320.SR", "1321.SR", "1322.SR", "2001.SR", "2010.SR", "2020.SR", "2060.SR", "2090.SR", 
    "2150.SR", "2170.SR", "2180.SR", "2200.SR", "2210.SR", "2220.SR", "2223.SR", "2240.SR", "2250.SR", "2290.SR", 
    "2300.SR", "2310.SR", "2330.SR", "2350.SR", "2360.SR", "3001.SR", "3002.SR", "3003.SR", "3004.SR", "3005.SR", 
    "3007.SR", "3008.SR", "3010.SR", "3020.SR", "3030.SR", "3040.SR", "3050.SR", "3060.SR", "3080.SR", "3090.SR", 
    "3091.SR", "1212.SR", "1214.SR", "1302.SR", "1303.SR", "2040.SR", "2110.SR", "2160.SR", "2320.SR", "2370.SR", 
    "4110.SR", "4140.SR", "4141.SR", "4142.SR", "1831.SR", "1832.SR", "1833.SR", "4270.SR", "6004.SR", "2190.SR", 
    "4031.SR", "4040.SR", "4260.SR", "4261.SR", "1213.SR", "2130.SR", "2340.SR", "4011.SR", "4012.SR", "4180.SR", 
    "1810.SR", "1820.SR", "1830.SR", "4170.SR", "4290.SR", "4291.SR", "4292.SR", "6002.SR", "6012.SR", "6013.SR", 
    "6014.SR", "6015.SR", "4070.SR", "4071.SR", "4210.SR", "4003.SR", "4008.SR", "4050.SR", "4051.SR", "4190.SR", 
    "4191.SR", "4192.SR", "4240.SR", "4001.SR", "4006.SR", "4061.SR", "4160.SR", "4161.SR", "4162.SR", "4163.SR", 
    "4164.SR", "2050.SR", "2100.SR", "2270.SR", "2280.SR", "2281.SR", "2282.SR", "2283.SR", "4080.SR", "6001.SR", 
    "6010.SR", "6020.SR", "6040.SR", "6050.SR", "6060.SR", "6070.SR", "6090.SR", "2140.SR", "2230.SR", "4002.SR", 
    "4004.SR", "4005.SR", "4007.SR", "4009.SR", "4013.SR", "4014.SR", "2070.SR", "4015.SR", "1010.SR", "1020.SR", 
    "1030.SR", "1050.SR", "1060.SR", "1080.SR", "1120.SR", "1140.SR", "1150.SR", "1180.SR", "1111.SR", "1182.SR", 
    "1183.SR", "2120.SR", "4081.SR", "4082.SR", "4130.SR", "4280.SR", "8010.SR", "8012.SR", "8020.SR", "8030.SR", 
    "8040.SR", "8050.SR", "8060.SR", "8070.SR", "8100.SR", "8120.SR", "8150.SR", "8160.SR", "8170.SR", "8180.SR", 
    "8190.SR", "8200.SR", "8210.SR", "8230.SR", "8240.SR", "8250.SR", "8260.SR", "8270.SR", "8280.SR", "8300.SR", 
    "8310.SR", "8311.SR", "7200.SR", "7201.SR", "7202.SR", "7203.SR", "7204.SR", "7010.SR", "7020.SR", "7030.SR", 
    "7040.SR", "2080.SR", "2081.SR", "2082.SR", "2083.SR", "5110.SR", "4330.SR", "4331.SR", "4332.SR", "4333.SR", 
    "4334.SR", "4335.SR", "4336.SR", "4337.SR", "4338.SR", "4340.SR", "4342.SR", "4344.SR", "4345.SR", "4346.SR", 
    "4347.SR", "4348.SR", "4349.SR", "4020.SR", "4090.SR", "4100.SR", "4150.SR", "4220.SR", "4230.SR", "4250.SR", 
    "4300.SR", "4320.SR", "4321.SR", "4322.SR",
]


USA = [
    "MSFT", "AAPL", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "LLY", "AVGO", "JPM", "V", "WMT", "XOM", "UNH",
    "MA", "PG", "HD", "ORCL", "COST", "MRK", "CVX", "ABBV", "CRM", "NFLX", "AMD", "KO", "PEP", "GE", "LIN",
    "ADBE", "TMO", "DIS", "MCD", "TMUS", "ABT", "QCOM", "DHR", "AMAT", "INTU", "CMCSA", "UBER", "NOW", "COP",
    "UNP", "NKE", "PM", "MU", "ISRG", "SCHW"
]

UK = [
    "AZN", "LYG", "HSBC", "UL", "BP", "GSK", "STLA", "RELX", "DEO", "BTI", "NGG", "FERG", "HLN",
    "BCS", "CCEP", "NWG", "WTW", "PUK", "VOD", "APTV", "IHG", "CNHI", "NVT", "SNN", "ROIV", "PSO",
    "LBTYB", "LBTYK", "LBTYA", "JHG", "ATCOL", "CLVT", "ARBKL", "NOMD", "LIVN", "TRMD", "IMCR",
    "MANU", "AY", "DAVA", "VTEX", "VRNA", "AUTL", "CNTA", "GSM", "BCYC", "BSIG"
]

Australian = [
    "BHP", "CBA", "RIO", "CSL", "WBC", "WES", "WDS", "RMD", "TLS", "WOW", "ALL", "JHX", "COL", "SUN", "AMC",
    "ORG", "IAG", "FPH", "MIN", "CAR", "ASX", "SOL", "TLC", "APA", "PME", "BSL"
]

all_symbols = [
    "4310.SR", "2222.SR", "2380.SR", "2381.SR", "4030.SR", "4200.SR", "1201.SR", "1202.SR", "1210.SR", "1211.SR", 
    "1301.SR", "1304.SR", "1320.SR", "1321.SR", "1322.SR", "2001.SR", "2010.SR", "2020.SR", "2060.SR", "2090.SR", 
    "2150.SR", "2170.SR", "2180.SR", "2200.SR", "2210.SR", "2220.SR", "2223.SR", "2240.SR", "2250.SR", "2290.SR", 
    "2300.SR", "2310.SR", "2330.SR", "2350.SR", "2360.SR", "3001.SR", "3002.SR", "3003.SR", "3004.SR", "3005.SR", 
    "3007.SR", "3008.SR", "3010.SR", "3020.SR", "3030.SR", "3040.SR", "3050.SR", "3060.SR", "3080.SR", "3090.SR", 
    "3091.SR", "1212.SR", "1214.SR", "1302.SR", "1303.SR", "2040.SR", "2110.SR", "2160.SR", "2320.SR", "2370.SR", 
    "4110.SR", "4140.SR", "4141.SR", "4142.SR", "1831.SR", "1832.SR", "1833.SR", "4270.SR", "6004.SR", "2190.SR", 
    "4031.SR", "4040.SR", "4260.SR", "4261.SR", "1213.SR", "2130.SR", "2340.SR", "4011.SR", "4012.SR", "4180.SR", 
    "1810.SR", "1820.SR", "1830.SR", "4170.SR", "4290.SR", "4291.SR", "4292.SR", "6002.SR", "6012.SR", "6013.SR", 
    "6014.SR", "6015.SR", "4070.SR", "4071.SR", "4210.SR", "4003.SR", "4008.SR", "4050.SR", "4051.SR", "4190.SR", 
    "4191.SR", "4192.SR", "4240.SR", "4001.SR", "4006.SR", "4061.SR", "4160.SR", "4161.SR", "4162.SR", "4163.SR", 
    "4164.SR", "2050.SR", "2100.SR", "2270.SR", "2280.SR", "2281.SR", "2282.SR", "2283.SR", "4080.SR", "6001.SR", 
    "6010.SR", "6020.SR", "6040.SR", "6050.SR", "6060.SR", "6070.SR", "6090.SR", "2140.SR", "2230.SR", "4002.SR", 
    "4004.SR", "4005.SR", "4007.SR", "4009.SR", "4013.SR", "4014.SR", "2070.SR", "4015.SR", "1010.SR", "1020.SR", 
    "1030.SR", "1050.SR", "1060.SR", "1080.SR", "1120.SR", "1140.SR", "1150.SR", "1180.SR", "1111.SR", "1182.SR", 
    "1183.SR", "2120.SR", "4081.SR", "4082.SR", "4130.SR", "4280.SR", "8010.SR", "8012.SR", "8020.SR", "8030.SR", 
    "8040.SR", "8050.SR", "8060.SR", "8070.SR", "8100.SR", "8120.SR", "8150.SR", "8160.SR", "8170.SR", "8180.SR", 
    "8190.SR", "8200.SR", "8210.SR", "8230.SR", "8240.SR", "8250.SR", "8260.SR", "8270.SR", "8280.SR", "8300.SR", 
    "8310.SR", "8311.SR", "7200.SR", "7201.SR", "7202.SR", "7203.SR", "7204.SR", "7010.SR", "7020.SR", "7030.SR", 
    "7040.SR", "2080.SR", "2081.SR", "2082.SR", "2083.SR", "5110.SR", "4330.SR", "4331.SR", "4332.SR", "4333.SR", 
    "4334.SR", "4335.SR", "4336.SR", "4337.SR", "4338.SR", "4340.SR", "4342.SR", "4344.SR", "4345.SR", "4346.SR", 
    "4347.SR", "4348.SR", "4349.SR", "4020.SR", "4090.SR", "4100.SR", "4150.SR", "4220.SR", "4230.SR", "4250.SR", 
    "4300.SR", "4320.SR", "4321.SR", "4322.SR",

    "MSFT", "AAPL", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "LLY", "AVGO", "JPM", "V", "WMT", "XOM", "UNH",
    "MA", "PG", "HD", "ORCL", "COST", "MRK", "CVX", "ABBV", "CRM", "NFLX", "AMD", "KO", "PEP", "GE", "LIN",
    "ADBE", "TMO", "DIS", "MCD", "TMUS", "ABT", "QCOM", "DHR", "AMAT", "INTU", "CMCSA", "UBER", "NOW", "COP",
    "UNP", "NKE", "PM", "MU", "ISRG", "SCHW",

    "AZN", "LYG", "HSBC", "UL", "BP", "GSK", "STLA", "RELX", "DEO", "BTI", "NGG", "FERG", "HLN",
    "BCS", "CCEP", "NWG", "WTW", "PUK", "VOD", "APTV", "IHG", "CNHI", "NVT", "SNN", "ROIV", "PSO",
    "LBTYB", "LBTYK", "LBTYA", "JHG", "ATCOL", "CLVT", "ARBKL", "NOMD", "LIVN", "TRMD", "IMCR",
    "MANU", "AY", "DAVA", "VTEX", "VRNA", "AUTL", "CNTA", "GSM", "BCYC", "BSIG",

    "BHP", "CBA", "RIO", "CSL", "WBC", "WES", "WDS", "RMD", "TLS", "WOW", "ALL", "JHX", "COL", "SUN", "AMC",
    "ORG", "IAG", "FPH", "MIN", "CAR", "ASX", "SOL", "TLC", "APA", "PME", "BSL"
]

# Load quantized models
def load_quantized_models(model_dir="tflite_models"):
    quantized_models = {}
    for symbol in all_symbols:
        model_path = os.path.join(model_dir, f"{symbol}_model.tflite")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        quantized_models[symbol] = interpreter
    return quantized_models

trained_models = load_quantized_models()


#               api keys
#    W8QKMjxaTE1NbdMETeefIFcqWb0q1o2v
#    b6kJR5n17nY59RPwlblXlKie6QJf6isp
#    zQSoMyE77xkaXdOoQhlNEHFSGmuAjPmi

# Function to fetch data asynchronously
# async def fetch_data_async(symbol, api_key='zQSoMyE77xkaXdOoQhlNEHFSGmuAjPmi',
#                            base_url='https://financialmodelingprep.com/api/v3'):
#     endpoint = f'{base_url}/historical-price-full/{symbol}?apikey={api_key}'
#     async with aiohttp.ClientSession() as session:
#         async with session.get(endpoint) as response:
#             if response.status == 200:
#                 data = await response.json()
#                 historical_data = data['historical']
#                 data_list = [{'Date': item['date'], 'Open': item['open'], 'High': item['high'], 'Low': item['low'],
#                               'Close': item['close'], 'AdjClose': item['adjClose'], 'Volume': item['volume']} for item
#                              in historical_data]
#                 df = pd.DataFrame(data_list)
#                 df = df[::-1].reset_index(drop=True)

#                 # Store data into CSV
#                 csv_path = os.path.join("data", f"{symbol}_data.csv")
#                 df.to_csv(csv_path, index=False)
#                 return df
#             else:
#                 print(f'Failed to retrieve data for {symbol}')
#                 return None

async def fetch_data_async(symbol, api_key='rJbxLKBGH8LBfX8zWtgB8FczEIUpkUQ7',
                           base_url='https://financialmodelingprep.com/api/v3'):
    endpoint = f'{base_url}/historical-price-full/{symbol}?apikey={api_key}'
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    historical_data = data['historical']
                    data_list = [{'Date': item['date'], 'Open': item['open'], 'High': item['high'], 'Low': item['low'],
                                  'Close': item['close'], 'AdjClose': item['adjClose'], 'Volume': item['volume']} for item
                                 in historical_data]
                    df = pd.DataFrame(data_list)
                    df = df[::-1].reset_index(drop=True)
                    # Store data into CSV
                    csv_path = os.path.join("data", f"{symbol}_data.csv")
                    df.to_csv(csv_path, index=False)
                    return df
                except KeyError:
                    print(f'KeyError: "historical" key not found in response for symbol {symbol}')
                    return None
            else:
                print(f'Failed to retrieve data for {symbol}')
                return None



async def load_data(symbol):
    # Fetch data from the API regardless of whether the CSV file exists or not
    return await fetch_data_async(symbol)



async def fetch_all_data():
    fetched_data = await asyncio.gather(*[load_data(symbol) for symbol in all_symbols])
    return fetched_data


# create_technical_indicators function
def create_technical_indicators(data):
    if 'MA10' not in data.columns:
        data['MA10'] = ta.sma(data['Close'], length=10)
        data['MA50'] = ta.sma(data['Close'], length=50)
        data['EMAF'] = ta.ema(data.Close, length=20)
        data['EMAM'] = ta.ema(data.Close, length=100)
        data['EMAS'] = ta.ema(data.Close, length=150)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['MACD'], data['MACD_signal'], _ = ta.macd(data['Close'])
    return data


# predict_tomorrow_movement function for quantized model
def predict_tomorrow_movement(stock_symbol, interpreter, data):
    latest_data = create_technical_indicators(data)
    latest_features = ['Open', 'High', 'Low', 'AdjClose', 'MA10', 'MA50', 'RSI']
    latest_data = latest_data[latest_features][-30:]
    scaler = MinMaxScaler()
    latest_data = scaler.fit_transform(latest_data)
    latest_data = np.expand_dims(latest_data.astype(np.float32), axis=0)  # Ensure data is of type FLOAT32

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, latest_data)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    predicted_movement = "Up" if prediction > 0.5 else "Down"
    return predicted_movement, prediction[0][0]


# predict_tomorrow_movement_for_symbols function for quantized
async def predict_tomorrow_movement_for_symbols(trained_models):
    predictions = {}
    for symbol, interpreter in trained_models.items():
        csv_path = os.path.join("data", f"{symbol}_data.csv")
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            prediction, probability = predict_tomorrow_movement(symbol, interpreter, data)
            predictions[symbol] = prediction, probability
    return predictions



def rank_stocks(predictions):
    ranked_stocks = sorted(predictions.items(), key=lambda x: x[1][1], reverse=True)
    return ranked_stocks  


def get_top_10_investment_opportunities(ranked_stocks):
    return ranked_stocks[:10]


@app.route('/ranked_stocks', methods=['GET'])
async def get_ranked_stocks():
    predictions = await predict_tomorrow_movement_for_symbols(trained_models)
    ranked_stocks = rank_stocks(predictions)
    ranked_stocks_json = [(symbol, float(probability)) for symbol, (prediction, probability) in ranked_stocks]
    return jsonify({'ranked_stocks': ranked_stocks_json})


# predict_all function
@app.route('/predict_all', methods=['GET'])
async def predict_all():
    predictions = await predict_tomorrow_movement_for_symbols(trained_models)
    # predictions = predict_tomorrow_movement_for_symbols(trained_models, data_frames)
    predictions_json = {symbol: {'prediction': prediction, 'probability': float(probability)} for
                        symbol, (prediction, probability) in predictions.items()}
    return jsonify({'category': 'All','predictions': predictions_json})


@app.route('/predict', methods=['GET'])
async def predict():
    category = request.args.get('category')
    if  category == 'usa':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await predict_tomorrow_movement_for_symbols(trained_models)).items() if symbol in USA}
    elif category == 'uk':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await predict_tomorrow_movement_for_symbols(trained_models)).items() if symbol in UK}
    elif category == 'australian':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await predict_tomorrow_movement_for_symbols(trained_models)).items() if symbol in Australian}
    elif category == 'saudia':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await predict_tomorrow_movement_for_symbols(trained_models)).items() if symbol in saudia}

    else : 
        predictions = await predict_tomorrow_movement_for_symbols(trained_models)
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for
                        symbol, (prediction, probability) in predictions.items()}

    response = {'category': category.capitalize() if category else 'All','predictions': predictions}
    return jsonify(response)


@app.route('/predict_stock', methods=['GET'])
async def predict_stock():
    stock_symbol = request.args.get('symbol')
    
    if stock_symbol is None or stock_symbol not in all_symbols:
        return jsonify({'error': 'Invalid stock symbol'}), 400

    csv_path = os.path.join("data", f"{stock_symbol}_data.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Data not available for the specified stock'}), 404

    data = pd.read_csv(csv_path)

    prediction, probability = predict_tomorrow_movement(stock_symbol, trained_models[stock_symbol], data)

    return jsonify({'symbol': stock_symbol, 'prediction': prediction, 'probability': float(probability)})


@app.route('/top_ten', methods=['GET'])
async def get_top_ten():
    category = request.args.get('category')
    predictions = await predict_tomorrow_movement_for_symbols(trained_models)

    if category == 'usa':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in USA}
    elif category == 'uk':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in UK}
    elif category == 'australian':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in Australian}
    elif category == 'saudia':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in saudia}
    else:
        predictions = predictions

    ranked_stocks = rank_stocks(predictions)
    top_ten = get_top_10_investment_opportunities(ranked_stocks)
    top_ten_json = [(symbol, float(probability)) for symbol, (prediction, probability) in top_ten]
    
    response = {'category': category.capitalize() if category else 'All', 'top_ten': top_ten_json}
    return jsonify(response)


# Function to update data from API and store in CSV files
async def update_data():
    print("Updating data from API...")
    await fetch_all_data()
    print("Data updated.")

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(id='update_data', func=run_main, trigger=CronTrigger(hour=0, minute=0))
    scheduler.start()

def run_main():
    loop = asyncio.new_event_loop()  
    asyncio.set_event_loop(loop) 
    loop.run_until_complete(update_data()) 

if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(fetch_all_data())
    start_scheduler()
    app.run(debug=False)
