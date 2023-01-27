from flask import Flask, request, jsonify
from pickle import load
import pandas as pd
import numpy as np
import xgboost as xgb
# load the model
tipster_bob_1 = load(open('tipster_bob_1.pkl', 'rb'))
tipster_bob_2 = load(open('tipster_bob_2.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
app = Flask(__name__)


def formatData(row):

    headers = ['tipDate', 'tipster',
               'live', 'units', 'odds',
               'allCatTotalROI',
               'allCatTotalWinRate', 'allCatTotalNumOfTips',
               'marketCatTotalROI',
               'marketCatTotalWinRate', 'marketCatTotalNumOfTips',
               'homeTeamCatTotalROI',
               'homeTeamCatTotalWinRate', 'homeTeamCatTotalNumOfTips',
               'awayTeamCatTotalROI',
               'awayTeamCatTotalWinRate', 'awayTeamCatTotalNumOfTips',
               'compCatTotalROI',
               'compCatTotalWinRate', 'compCatTotalNumOfTips',
               'sportCatTotalROI',
               'sportCatTotalWinRate', 'sportCatTotalNumOfTips',
               'allCatTenROI',
               'allCatTenWinRate', 'allCatTenNumOfTips',
               'marketCatTenROI',
               'marketCatTenWinRate', 'marketCatTenNumOfTips',
               'homeTeamCatTenROI',
               'homeTeamCatTenWinRate', 'homeTeamCatTenNumOfTips',
               'awayTeamCatTenROI',
               'awayTeamCatTenWinRate', 'awayTeamCatTenNumOfTips',
               'compCatTenROI',
               'compCatTenWinRate', 'compCatTenNumOfTips',
               'sportCatTenROI',
               'sportCatTenWinRate', 'sportCatTenNumOfTips',
               'allCatFiveROI',
               'allCatFiveWinRate', 'allCatFiveNumOfTips',
               'marketCatFiveROI',
               'marketCatFiveWinRate', 'marketCatFiveNumOfTips',
               'homeTeamCatFiveROI',
               'homeTeamCatFiveWinRate', 'homeTeamCatFiveNumOfTips',
               'awayTeamCatFiveROI',
               'awayTeamCatFiveWinRate', 'awayTeamCatFiveNumOfTips',
               'compCatFiveROI',
               'compCatFiveWinRate', 'compCatFiveNumOfTips',
               'sportCatFiveROI',
               'sportCatFiveWinRate', 'sportCatFiveNumOfTips'
               ]

    # print(len(headers))
    dataList = []
    newlist = sorted([row], key=lambda d: d['tipDate'])
    i = 0
    for tip in newlist:
        # tipDate, tipsterId, live, units, odds, allCat, marketCat, homeTeamCat, awayTeamCat, compCat, sportCat
        rowList = [
            i,
            str(tip['tipster']),
            0 if tip['live'] is False else 1,
            tip['units'],
            tip['odds'],
            tip['allCat']['total']['roi'],
            tip['allCat']['total']['winrate'],
            tip['allCat']['total']['numOfTips'],
            tip['marketCat']['total']['roi'],
            tip['marketCat']['total']['winrate'],
            tip['marketCat']['total']['numOfTips'],
            tip['compCat']['total']['roi'],
            tip['compCat']['total']['winrate'],
            tip['compCat']['total']['numOfTips'],
            tip['sportCat']['total']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['roi'],
            tip['sportCat']['total']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['winrate'],
            tip['sportCat']['total']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['numOfTips'],
            tip['homeTeamCat']['total']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['roi'],
            tip['homeTeamCat']['total']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['winrate'],
            tip['homeTeamCat']['total']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['numOfTips'],
            tip['awayTeamCat']['total']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['roi'],
            tip['awayTeamCat']['total']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['winrate'],
            tip['awayTeamCat']['total']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['total']['numOfTips'],
            tip['allCat']['ten']['roi'],
            tip['allCat']['ten']['winrate'],
            tip['allCat']['ten']['numOfTips'],
            tip['marketCat']['ten']['roi'],
            tip['marketCat']['ten']['winrate'],
            tip['marketCat']['ten']['numOfTips'],
            tip['compCat']['ten']['roi'],
            tip['compCat']['ten']['winrate'],
            tip['compCat']['ten']['numOfTips'],
            tip['sportCat']['ten']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['roi'],
            tip['sportCat']['ten']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['winrate'],
            tip['sportCat']['ten']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['numOfTips'],
            tip['homeTeamCat']['ten']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['roi'],
            tip['homeTeamCat']['ten']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['winrate'],
            tip['homeTeamCat']['ten']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['numOfTips'],
            tip['awayTeamCat']['ten']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['roi'],
            tip['awayTeamCat']['ten']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['winrate'],
            tip['awayTeamCat']['ten']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['ten']['numOfTips'],
            tip['allCat']['five']['roi'],
            tip['allCat']['five']['winrate'],
            tip['allCat']['five']['numOfTips'],
            tip['marketCat']['five']['roi'],
            tip['marketCat']['five']['winrate'],
            tip['marketCat']['five']['numOfTips'],
            tip['compCat']['five']['roi'],
            tip['compCat']['five']['winrate'],
            tip['compCat']['five']['numOfTips'],
            tip['sportCat']['five']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['roi'],
            tip['sportCat']['five']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['winrate'],
            tip['sportCat']['five']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['numOfTips'],
            tip['homeTeamCat']['five']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['roi'],
            tip['homeTeamCat']['five']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['winrate'],
            tip['homeTeamCat']['five']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['numOfTips'],
            tip['awayTeamCat']['five']['roi'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['roi'],
            tip['awayTeamCat']['five']['winrate'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['winrate'],
            tip['awayTeamCat']['five']['numOfTips'] if tip['compCat']['tag'].lower(
            ) != "combo" else tip['compCat']['five']['numOfTips']
        ]
        # print(len(rowList))
        dataList.append(rowList)
        i = i+1
    print(dataList)
    formattedData = pd.DataFrame(dataList, columns=headers)
    formattedData.sort_values(
        'tipDate', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')

    # print(formattedData['tipster'])
    # creating a list containing every tipster
    tipsters = list(set(formattedData['tipster'].values))
    tipsterToIdx = {t: i for i, t in enumerate(tipsters)}
    # print(tipsterToIdx)
    # assigning the tipster their corresponding tipster id
    tipsterId = [tipsterToIdx[id]
                 for id in list(formattedData['tipster'].values)]

    # creating a new column for the home tipster id
    formattedData['tipsterId'] = tipsterId
    # print(formattedData['tipsterId'])
    formattedData = formattedData.replace([np.inf, -np.inf, -0], 0)

    X_all = formattedData.drop(['tipster'], 1)

    X_all = scaler.fit_transform(np.asarray(X_all))

    return xgb.DMatrix(X_all)


@app.route('/prediction', methods=('POST',))
def prediction():
    content = request.json()
    print(content)
    formatted_data = formatData(content['row'])
    print(formatted_data)
    prediction_a = tipster_bob_1.predict(formatted_data)
    prediction_b = tipster_bob_2.predict(formatted_data)
    print(prediction)
    return jsonify({"predictions": {"prediction_a": prediction_a, "prediction_b": prediction_b}})


app.run('0.0.0.0', debug=True, port=8100)
