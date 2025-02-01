import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, redirect, request, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from scipy import stats
from werkzeug.utils import secure_filename

from data import db_session
from data.users import User

app = Flask(__name__)

app.config['SECRET_KEY'] = 'you_know_its_a_secret_key_really_secret'

# Конфигурация для загрузки файлов
UPLOAD_FOLDER = 'uploads'  # Папка для сохранения файлов
ALLOWED_EXTENSIONS = {'csv', 'json'}  # Разрешенные расширения файлов
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создаем папку для загрузок, если она не существует
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)


def allowed_file(filename):
    """Проверяем, что файл имеет допустимое расширение."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@login_manager.user_loader
def load_user(user_id):  # загрузка пользователя (flask_login)
    db_sess = db_session.create_session()
    return db_sess.query(User).get(user_id)


@app.route('/')  # основная страница
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        db_sess = db_session.create_session()
        if db_sess.query(User).filter(User.email == email).first():
            return jsonify({"message": "Такой пользователь уже есть!"}), 409
        user = User(name=name, email=email)
        user.set_password(password)
        db_sess.add(user)
        db_sess.commit()
        login_user(user)
        return jsonify({"message": "Регистрация прошла успешно!"}), 200


@app.route('/login', methods=['GET', 'POST'])  # страница для входа в систему
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')
        rememberMe = data.get('rememberMe')
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.email == email).first()
        if user and user.check_password(password):
            login_user(user, remember=rememberMe)
            return jsonify({"message": "Вход выполнен успешно!"}), 200
        return jsonify({"message": "Неправильный логин или пароль!"}), 401


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")


@app.route('/main', methods=['GET'])
def main_page():
    return render_template('main.html', name=current_user.name)


@app.route('/generate_plot', methods=['POST'])
@login_required
def generate_plot():
    try:
        data = request.json
        title = data.get('title')
        graphType = data.get('graphType')  # Тип графика

        # Получаем текущего пользователя и его датасет
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.id == current_user.id).first()

        if not user or not user.dataset_name:
            return jsonify({'error': 'Датасет пользователя не найден'}), 404

        # Загружаем датасет пользователя
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.dataset_name)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл датасета не найден'}), 404

        df = pd.read_csv(filepath)

        # Генерация данных в зависимости от типа графика
        if graphType == 'scatterplot' or graphType == 'lineplot':
            xAxis = data.get('xAxis')
            yAxis = data.get('yAxis')
            group = data.get('group')  # Колонка для группировки

            # Проверяем, что указанные колонки существуют в датасете
            if xAxis not in df.columns or yAxis not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400

            # Заменяем NaN на None (null в JSON)
            df[xAxis] = df[xAxis].replace({np.nan: None})
            df[yAxis] = df[yAxis].replace({np.nan: None})

            traces = []
            if group and group in df.columns:  # Если указана группировка
                grouped = df.groupby(group)
                for name, group_data in grouped:
                    if graphType == 'scatterplot':
                        trace = go.Scatter(
                            x=group_data[xAxis].tolist(),
                            y=group_data[yAxis].tolist(),
                            mode='markers',
                            name=str(name),  # Название группы
                        )
                    else:  # lineplot
                        trace = go.Scatter(
                            x=group_data[xAxis].tolist(),
                            y=group_data[yAxis].tolist(),
                            mode='lines',
                            name=str(name),  # Название группы
                        )
                    traces.append(trace)
            else:  # Если группировка не указана
                if graphType == 'scatterplot':
                    trace = go.Scatter(
                        x=df[xAxis].tolist(),
                        y=df[yAxis].tolist(),
                        mode='markers',
                        name=title,
                    )
                else:  # lineplot
                    trace = go.Scatter(
                        x=df[xAxis].tolist(),
                        y=df[yAxis].tolist(),
                        mode='lines',
                        name=title,
                    )
                traces.append(trace)

            # Формируем макет графика с осями
            layout = go.Layout(
                title=title,
                xaxis={'title': xAxis},
                yaxis={'title': yAxis},
            )

        elif graphType == 'boxplot' or graphType == 'histogram':
            values = data.get('values')
            group = data.get('group')  # Колонка для группировки

            # Проверяем, что указанная колонка существует в датасете
            if values not in df.columns:
                return jsonify({'error': 'Указанная колонка не найдена в датасете'}), 400

            # Заменяем NaN на None (null в JSON)
            df[values] = df[values].replace({np.nan: None})

            traces = []
            if group and group in df.columns:  # Если указана группировка
                grouped = df.groupby(group)
                for name, group_data in grouped:
                    if graphType == 'boxplot':
                        trace = go.Box(
                            y=group_data[values].tolist(),
                            name=str(name),  # Название группы
                        )
                    else:  # histogram
                        trace = go.Histogram(
                            x=group_data[values].tolist(),
                            name=str(name),  # Название группы
                        )
                    traces.append(trace)
            else:  # Если группировка не указана
                if graphType == 'boxplot':
                    trace = go.Box(
                        y=df[values].tolist(),
                        name=title,
                    )
                else:  # histogram
                    trace = go.Histogram(
                        x=df[values].tolist(),
                        name=title,
                    )
                traces.append(trace)

            # Формируем макет графика с осью Y (для boxplot и histogram)
            layout = go.Layout(
                title=title,
                yaxis={'title': values},
            )

        elif graphType == 'piechart':
            values = data.get('values')
            try:
                hole = float(data.get('hole')) if data.get('hole') else 0.0  # Если hole пустое, используем 0.0
            except ValueError:
                return jsonify({'error': 'Некорректное значение для параметра hole'}), 400
            if not (0 <= hole <= 1):
                return jsonify({'error': 'Параметр hole должен быть в диапазоне от 0 до 1'}), 400

            # Проверяем, что указанная колонка существует в датасете
            if values not in df.columns:
                return jsonify({'error': 'Указанная колонка не найдена в датасете'}), 400

            # Заменяем NaN на None (null в JSON)
            df[values] = df[values].replace({np.nan: None})
            value_counts = df[values].value_counts().reset_index()
            value_counts.columns = ['value', 'count']

            trace = go.Pie(
                labels=value_counts['value'].tolist(),
                values=value_counts['count'].tolist(),
                name=title,
                hole=hole,
            )

            # Формируем макет графика только с заголовком
            layout = go.Layout(
                title=title,
            )
            traces = [trace]

        elif graphType == 'barchart':
            values = data.get('values')
            orientation = data.get('orientation', 'v')  # По умолчанию вертикальная ориентация
            group = data.get('group')  # Колонка для группировки

            # Проверяем, что указанная колонка существует в датасете
            if values not in df.columns:
                return jsonify({'error': 'Указанная колонка не найдена в датасете'}), 400

            # Заменяем NaN на None (null в JSON)
            df[values] = df[values].replace({np.nan: None})

            traces = []
            if group and group in df.columns:  # Если указана группировка
                grouped = df.groupby(group)
                for name, group_data in grouped:
                    value_counts = group_data[values].value_counts().reset_index()
                    value_counts.columns = ['value', 'count']

                    if orientation == 'v':
                        # Вертикальная ориентация: x - уникальные значения, y - их количество
                        x_data = value_counts['value'].tolist()
                        y_data = value_counts['count'].tolist()
                    else:
                        # Горизонтальная ориентация: y - уникальные значения, x - их количество
                        x_data = value_counts['count'].tolist()
                        y_data = value_counts['value'].tolist()

                    trace = go.Bar(
                        x=x_data,
                        y=y_data,
                        name=str(name),  # Название группы
                        orientation=orientation,
                    )
                    traces.append(trace)
            else:  # Если группировка не указана
                value_counts = df[values].value_counts().reset_index()
                value_counts.columns = ['value', 'count']

                if orientation == 'v':
                    x_data = value_counts['value'].tolist()
                    y_data = value_counts['count'].tolist()
                else:
                    x_data = value_counts['count'].tolist()
                    y_data = value_counts['value'].tolist()

                trace = go.Bar(
                    x=x_data,
                    y=y_data,
                    name=title,
                    orientation=orientation,
                )
                traces.append(trace)

            # Формируем макет графика с осью X или Y в зависимости от ориентации
            layout = go.Layout(
                title=title,
                xaxis={'title': 'Количество' if orientation == 'v' else 'Значения'},
                yaxis={'title': 'Значения' if orientation == 'v' else 'Количество'},
            )

        else:
            return jsonify({'error': 'Неизвестный тип графика'}), 400

        # Преобразуем объекты Plotly в словари
        traces_dict = [trace.to_plotly_json() for trace in traces]
        layout_dict = layout.to_plotly_json()

        # Возвращаем JSON для Plotly
        return jsonify({
            'data': traces_dict,  # Данные для графика
            'layout': layout_dict  # Макет графика
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Возвращаем ошибку в формате JSON


@app.route('/run_test', methods=['POST'])
def run_test():
    try:
        # Получаем текущего пользователя и его датасет
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.id == current_user.id).first()

        if not user or not user.dataset_name:
            return jsonify({'error': 'Датасет пользователя не найден'}), 404

        # Загружаем датасет пользователя
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.dataset_name)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл датасета не найден'}), 404

        df = pd.read_csv(filepath)
        data = request.get_json()
        test_type = data.get('type')

        if test_type == "t-test Стьюдента":
            values_column = data.get('values')
            group_column = data.get('group')
            if values_column not in df.columns or group_column not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            # Проверка на две группы
            df1 = df.dropna(subset=[values_column, group_column])
            groups = df1[group_column].unique()
            if len(groups) != 2:
                return jsonify({"error": "t-test требует ровно две группы"}), 400

            group1 = df1[df1[group_column] == groups[0]][values_column]
            group2 = df1[df1[group_column] == groups[1]][values_column]

            # Проведение t-теста
            t_stat, p_value = stats.ttest_ind(group1, group2)
            result = {
                "statistic": t_stat,
                "p_value": p_value,
                "null_hypothesis": "Нулевая гипотеза отвергается" if p_value < 0.05 else "Нулевая гипотеза не отвергается",
                "result": "Средние значения двух групп равны" if p_value >= 0.05 else "Средние значения двух групп не равны"
            }

        elif test_type == "U-критерий Манна-Уитни":
            values_column = data.get('values')
            group_column = data.get('group')
            if values_column not in df.columns or group_column not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            df1 = df.dropna(subset=[values_column, group_column])
            # Проверка на две группы
            groups = df1[group_column].unique()
            if len(groups) != 2:
                return jsonify({"error": "U-критерий требует ровно две группы"}), 400

            group1 = df1[df1[group_column] == groups[0]][values_column]
            group2 = df1[df1[group_column] == groups[1]][values_column]

            # Проведение U-теста
            u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            result = {
                "statistic": u_stat,
                "p_value": p_value,
                "null_hypothesis": "Нулевая гипотеза отвергается" if p_value < 0.05 else "Нулевая гипотеза не отвергается",
                "result": "Распределения двух групп одинаковы" if p_value >= 0.05 else "Распределения двух групп не одинаковы"
            }

        elif test_type == "Тест Шапиро-Уилка":
            values_column = data.get('values')
            if values_column not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            df1 = df.dropna(subset=[values_column])
            # Проведение теста Шапиро-Уилка
            w_stat, p_value = stats.shapiro(df1[values_column])
            result = {
                "statistic": w_stat,
                "p_value": p_value,
                "null_hypothesis": "Нулевая гипотеза отвергается" if p_value < 0.05 else "Нулевая гипотеза не отвергается",
                "result": "Данные имеют нормальное распределение" if p_value >= 0.05 else 'Данные имеют ненормальное распределение'
            }
        elif test_type == "Коэффициент корреляции Пирсона":
            values_1 = data.get('values_1')
            values_2 = data.get('values_2')
            if values_1 not in df.columns or values_2 not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            # Проверка на две группы
            df1 = df.dropna(subset=[values_2, values_1])
            # Проведение t-теста
            stat, p_value = stats.pearsonr(df1[values_1], df1[values_2])
            result = {
                "statistic": stat,
                "p_value": p_value,
                "null_hypothesis": "Линейная взаимосвязь есть (но обратите внимание на коэффициент корреляции)" if p_value < 0.05 else "Линейной взаимосвязи не наблюдается (тем не менее, обратите внимание на коэффициент корреляции). p-value больше 0.05 может быть вызван размером выборки и другими факторами",
                "result": f"Коэффициент корреляции {stat}"
            }
        elif test_type == "Коэффициент корреляции Спирмена":
            values_1 = data.get('values_1')
            values_2 = data.get('values_2')
            if values_1 not in df.columns or values_2 not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            # Проверка на две группы
            df1 = df.dropna(subset=[values_2, values_1])
            # Проведение t-теста
            stat, p_value = stats.spearmanr(df1[values_1], df1[values_2])
            result = {
                "statistic": stat,
                "p_value": p_value,
                "null_hypothesis": "Линейная взаимосвязь есть (но обратите внимание на коэффициент корреляции)" if p_value < 0.05 else "Линейной взаимосвязи не наблюдается (тем не менее, обратите внимание на коэффициент корреляции). p-value больше 0.05 может быть вызван размером выборки и другими факторами",
                "result": f"Коэффициент корреляции {stat}"
            }
        elif test_type == "ANOVA":
            values_column = data.get('values')
            group_column = data.get('group')
            if values_column not in df.columns or group_column not in df.columns:
                return jsonify({'error': 'Указанные колонки не найдены в датасете'}), 400
            # Проверка на две группы
            df1 = df.dropna(subset=[values_column, group_column])
            groups = df1[group_column].unique()

            g = []
            for i in groups:
                x = df1[df1[group_column] == i][values_column]
                g.append(x)

            # Проведение t-теста
            stat, p_value = stats.f_oneway(*g)
            result = {
                "statistic": stat,
                "p_value": p_value,
                "null_hypothesis": "Нулевая гипотеза отвергается" if p_value < 0.05 else "Нулевая гипотеза не отвергается",
                "result": "Средние значения групп равны" if p_value >= 0.05 else "Средние значения групп не равны"
            }
        else:
            return jsonify({"error": "Неизвестный тип теста"}), 400
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": f"Колонка {str(e)} не найдена в датасете"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        # Проверяем, есть ли файл в запросе
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден в запросе'}), 400

        file = request.files['file']

        # Проверяем, что файл имеет допустимое расширение
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Неподдерживаемый формат файла. Используйте CSV или JSON.'}), 400

        # Получаем параметры из формы
        delimiter = request.form.get('delimiter', ',')  # По умолчанию запятая
        decimal = request.form.get('decimal', '.')      # По умолчанию точка

        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filename = filename.rsplit('.', 1)[0] + '.csv'  # Меняем расширение на .csv
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Читаем файл в зависимости от его формата
        if file.filename.endswith('.csv'):
            # Читаем CSV-файл с учетом разделителей
            df = pd.read_csv(file, sep=delimiter, decimal=decimal)
        elif file.filename.endswith('.json'):
            # Читаем JSON-файл
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Неподдерживаемый формат файла. Используйте CSV или JSON.'}), 400

        # Сохраняем файл в формате CSV с указанными параметрами
        df.to_csv(filepath, index=False, sep=',', decimal='.')

        # Обновляем информацию о файле в базе данных
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.id == current_user.id).first()

        # Удаляем старый файл, если он существует
        if user.dataset_name is not None:
            old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.dataset_name)
            if os.path.exists(old_filepath):
                try:
                    os.remove(old_filepath)
                except Exception as e:
                    pass  # Игнорируем ошибки при удалении старого файла

        # Сохраняем новое имя файла в базе данных
        user.dataset_name = filename
        db_sess.commit()

        # Возвращаем информацию о загруженном файле
        return jsonify({
            'message': 'Файл успешно загружен',
            'filename': filename,
            'filepath': filepath,
            'uploaded_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Эндпоинт для получения первых 5 строк
@app.route('/get_first_rows', methods=['GET'])
def get_first_rows():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename is required'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Чтение файла (поддерживаем CSV и JSON)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, sep=',')
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Возвращаем первые 5 строк в виде списка словарей
        first_rows = df.head().to_json(orient='records', default_handler=str)
        return jsonify({'data': first_rows}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    db_session.global_init('db/plotter.db')
    app.run(debug=True, port=os.getenv("PORT", default=5000))
