from flask import Flask, redirect, render_template, request, session, url_for
import os
import sqlite3 as sl
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)
db = "world_population.db"


@app.route("/")
def home():
    er = session.pop('login_error', None)
    if er is None:
        return render_template("index.html", login_error='')
    else:
        return render_template("index.html", login_error=er)


@app.route("/client")
def client():
    logged_in = session.pop("logged_in", None)
    if logged_in is None:
        session["logged_in"] = False
        session['login_error'] = 'you are not logged in.'
        return redirect(url_for("home"))
    else:
        session["logged_in"] = True
        l = csv_create_country_list()
        return render_template('home.html', country_list=l)


@app.route('/result')
def result():
    c_name = session.pop('country', None)
    if c_name is None:
        return redirect(url_for('client'))
    else:
        return render_template('result.html', c_name=c_name)


@app.route("/select", methods=["POST", "GET"])
def select():
    if request.method == "POST":
        session["country"] = request.form['country']
        pred, acc, nyl, ncl = csv_ml(request.form['country'], request.form['year'])
        csv_create_graph(request.form['country'], nyl, ncl)
        return redirect(url_for('result'))


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST" and db_check_creds(request.form["username"], request.form["password"]):
        session["username"] = request.form["username"]
        session["logged_in"] = True
        return redirect(url_for('client'))
    else:
        session['login_error'] = 'login failed.'
        return redirect(url_for('home'))


@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == "POST":
        db_create_user(request.form['username'], request.form['password'])
        session["username"] = request.form["username"]
        session["logged_in"] = True
        return redirect(url_for('client'))


@app.route("/logout", methods=["POST", "GET"])
def logout():
    # destroy session
    # send them back to login page
    if request.method == "POST":
        session["logged_in"] = False
        session.pop('username', None)
        return redirect(url_for('home'))


@app.route("/back", methods=["POST", "GET"])
def back():
    return redirect(url_for('client'))


def db_create_db():
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt = "CREATE TABLE IF NOT EXISTS credentials(id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(20) NOT " \
           "NULL, password VARCHAR(20) NOT NULL); "
    curs.execute(stmt)
    conn.close()


def db_create_user(un, pw):
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (un, pw)
    stmt = "INSERT OR IGNORE INTO credentials (username, password) VALUES " + str(v)
    curs.execute(stmt)
    conn.commit()
    conn.close()


def db_check_creds(un, pw):
    conn = sl.connect(db)
    curs = conn.cursor()
    v = (un,)
    stmt = 'SELECT * FROM credentials WHERE username=?'
    curs.execute(stmt, v)
    if pw == curs.fetchone()[2]:
        conn.close()
        return True
    conn.close()
    return False


def csv_create_country_list():
    df = pd.read_csv('Population.csv')
    return list(df.columns)[1:]


def csv_ml(country, year_input):
    df = pd.read_csv('Population.csv')
    df = df.dropna()

    x = df[['Year']].values
    y = df[country].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    print('Train Set:', x_train.shape, y_train.shape)
    print('Test Set:', x_test.shape, y_test.shape)

    knn = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    knn.fit(x_train, y_train)

    x_new = np.array([[year_input]])
    prediction = knn.predict(x_new)
    prediction = int(prediction)
    accuracy = knn.score(x_test, y_test)

    new_year_list = list(df['Year']) + [year_input]
    new_country_list = list(df[country]) + [prediction]

    return prediction, accuracy, new_year_list, new_country_list


def csv_create_graph(country_input, new_year_list, new_country_list):
    df = pd.read_csv('Population.csv')
    df = df.dropna()

    label = 'Existing Population Data of ' + country_input
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].plot(df['Year'], df[country_input], label=label)

    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Population (k)')
    ax[0].set_title('Population of ' + country_input + ' over the Years')
    ax[0].legend()

    label = 'Predictive Population Data of ' + country_input
    ax[1].plot(new_year_list, new_country_list, label=label)
    # set upper subplot characteristics
    ax[1].set_xlabel('Year')
    ax[1].set_ylabel('Population (k)')
    ax[1].set_title('Population of ' + country_input + ' in Future Years')
    ax[1].legend()

    df['Mean'] = df.mean(axis=1)
    ax[2].plot(df['Year'], df['Mean'], label='Average')
    ax[2].set_xlabel('Year')
    ax[2].set_ylabel('Population (k)')
    ax[2].set_title('Average of Population across All Countries')
    ax[2].legend()

    plt.savefig('static/result.png')


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)


    # user1, password1