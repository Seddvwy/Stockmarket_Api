from flask import Flask, jsonify, request
import utils
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/ranked_stocks', methods=['GET'])
async def get_ranked_stocks():
    predictions = await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)
    ranked_stocks = utils.rank_stocks(predictions)
    ranked_stocks_json = [(symbol, float(probability)) for symbol, (prediction, probability) in ranked_stocks]
    return jsonify({'ranked_stocks': ranked_stocks_json})


# predict_all function
@app.route('/predict_all', methods=['GET'])
async def predict_all():
    predictions = await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)
    # predictions = predict_tomorrow_movement_for_symbols(trained_models, data_frames)
    predictions_json = {symbol: {'prediction': prediction, 'probability': float(probability)} for
                        symbol, (prediction, probability) in predictions.items()}
    return jsonify({'category': 'All','predictions': predictions_json})


@app.route('/predict', methods=['GET'])
async def predict():
    category = request.args.get('category')
    if  category == 'usa':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)).items() if symbol in utils.USA}
    elif category == 'uk':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)).items() if symbol in utils.UK}
    elif category == 'aus':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)).items() if symbol in utils.AUS}
    elif category == 'sau':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)).items() if symbol in utils.SAU}
    elif category == 'jap':
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for symbol, (prediction, probability) in (await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)).items() if symbol in utils.JAP}


    else : 
        predictions = await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)
        predictions = {symbol: {'prediction': prediction, 'probability': float(probability)} for
                        symbol, (prediction, probability) in predictions.items()}

    response = {'category': category.capitalize() if category else 'All','predictions': predictions}
    return jsonify(response)


@app.route('/predict_stock', methods=['GET'])
async def predict_stock():
    stock_symbol = request.args.get('symbol')
    
    if stock_symbol is None or stock_symbol not in utils.all_symbols:
        return jsonify({'error': 'Invalid stock symbol'}), 400

    csv_path = os.path.join("data", f"{stock_symbol}_data.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Data not available for the specified stock'}), 404

    data = pd.read_csv(csv_path)

    prediction, probability = utils.predict_tomorrow_movement(stock_symbol, utils.trained_models[stock_symbol], data)

    return jsonify({'symbol': stock_symbol, 'prediction': prediction, 'probability': float(probability)})


@app.route('/top_ten', methods=['GET'])
async def get_top_ten():
    category = request.args.get('category')
    predictions = await utils.predict_tomorrow_movement_for_symbols(utils.trained_models)

    if category == 'usa':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in utils.USA}
    elif category == 'uk':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in utils.UK}
    elif category == 'aus':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in utils.AUS}
    elif category == 'sau':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in utils.SAU}
    elif category == 'jap':
        predictions = {symbol: prediction for symbol, prediction in predictions.items() if symbol in utils.JAP}    
    else:
        predictions = predictions

    ranked_stocks = utils.rank_stocks(predictions)
    top_ten = utils.get_top_10_investment_opportunities(ranked_stocks)
    top_ten_json = [(symbol, float(probability)) for symbol, (prediction, probability) in top_ten]
    
    response = {'category': category.capitalize() if category else 'All', 'top_ten': top_ten_json}
    return jsonify(response)



if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(fetch_all_data())
    utils.start_scheduler()
    app.run(debug=False)
