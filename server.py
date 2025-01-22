from data import db_session
import os
from flask import Flask, render_template, redirect, request, abort, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from data.users import User
from Forms.forms import *


UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you_know_its_a_secret_key_really_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)


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


@app.route('/logout')  # выход пользователя (flask_login) ЕЩЕ НЕТ КНОПКИ ДЛЯ ВЫХОДА
@login_required
def logout():
    logout_user()
    return redirect("/")

'''
@app.route('/change_profile', methods=['POST', 'GET'])  # изменение профиля
@login_required
def change():
    db_sess = db_session.create_session()
    user = db_sess.query(User).filter(User.id == current_user.id).first()
    if not(user):
        abort(520)
    if len(request.args['name']) > 0:
        user.name = request.args['name']
    if len(request.args['surname']) > 0:
        user.surname = request.args['surname']
    if len(request.args['school']) > 0:
        user.school = request.args['school']
    if request.args['position'] != 'Не выбрано':
        user.position = request.args['position']
    if len(request.args['classClub']) > 0:
        user.classClub = request.args['classClub']
    if len(request.args['email']) > 0:
        user.email = request.args['email']
    db_sess.commit()
    return redirect('/profile')


@app.route('/send_request', methods=['POST', 'GET'])  # отправка профиля на модерацию
@login_required
def send_request():
    db_sess = db_session.create_session()
    user = db_sess.query(User).filter(User.id == current_user.id).first()
    if user.verified == 0:
        return render_template('profile.html', message='Заявка уже была подана')
    if user.school == None:
        return render_template('profile.html', message='Не указана школа')
    if user.position == 'Учитель' or user.position == 'Ученик':
        if user.classClub == None:
            return render_template('profile.html', message='Не указан класс')
        user.verified = 0
        db_sess.commit()
        return render_template('profile.html', message='Заявка подана')
    elif user.position == 'Админ школы':
        user.verified = 0
        db_sess.commit()
        return render_template('profile.html', message='Заявка подана')
    return render_template('profile.html', message='Не выбрана роль')


@app.route('/profile')  # страница с профилем
@login_required
def profile():
    return render_template('profile.html', title='Профиль')


@app.route('/requests', methods=['GET', 'POST'])  # список профилей на модерации
@login_required
def requests():
    db_sess = db_session.create_session()
    if current_user.position == 'Учитель':
        users = db_sess.query(User).filter(User.position == 'Ученик', User.classClub == current_user.classClub,
                                           User.verified == 0, User.school == current_user.school).all()
    elif current_user.position == 'Админ школы':
        users = db_sess.query(User).filter((User.position == 'Учитель') | (User.position == 'Ученик') | (User.position == 'Админ школы'),
                                           User.verified == 0, User.school == current_user.school).all()
    else:
        abort(403)
    return render_template('requests.html', title='Заявки', users=users)


@app.route('/schools', methods=['GET', 'POST'])  # страница для добавления школы
@login_required
def schools():
    form = AddSchool()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        school = db_sess.query(School).filter(School.title == form.name.data).first()
        admin = db_sess.query(User).filter(User.id == form.idAdmin.data).first()
        if school or not(admin):
            return render_template('school.html', form=form, title='Добавление школы')
        if admin.position != 'Админ школы':
            return render_template('school.html', form=form, title='Добавление школы')
        school = School(title=form.name.data)
        db_sess.add(school)
        school = db_sess.query(School).filter(School.title == school.title).first()
        admin.school = school.id
        admin.verified = 2
        db_sess.commit()
        return redirect('/')
    return render_template('school.html', title='Добавление школы', form=form)


@app.route('/accept_request/<int:id>')  # принять заявку на модерации
@login_required
def accept_request(id):
    db_sess = db_session.create_session()
    user = db_sess.query(User).filter(User.id == id).first()
    user.verified = 2
    db_sess.commit()
    return redirect('/requests')


@app.route('/cancel_request/<int:id>')  # отклонить заявку на модерации
@login_required
def cancel_request(id):
    db_sess = db_session.create_session()
    cancel = request.args['cancel']
    user = db_sess.query(User).filter(User.id == id).first()
    user.whyCancelled = cancel
    user.verified = 1
    db_sess.commit()
    return redirect('/requests')


@app.route('/lostList')  # список потерянных вещей
@login_required
def lostList():
    db_sess = db_session.create_session()
    lost = db_sess.query(Lost).filter(Lost.school == current_user.school).order_by(Lost.userFound != current_user.id).all()
    return render_template('lostList.html', lost=lost, title='Список потеряшек')


@app.route('/lost/<int:id>')  # потерянная вещь (подробно)
@login_required
def lost(id):
    db_sess = db_session.create_session()
    lost = db_sess.query(Lost).filter(Lost.school == current_user.school, Lost.id == id).first()
    return render_template('lost.html', lost=lost, title='Потеряшка')


@app.route('/addLostForm', methods=['GET', 'POST'])  # страница для добавления потерянной вещи
@login_required
def addLostForm():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('lostForm.html', message='Файл не выбран', title='добавление потеряшки')
        file = request.files['file']
        if file.filename == '':
            return render_template('lostForm.html', message='Файл не выбран', title='добавление потеряшки')
        if not(allowed_file(file.filename)):
            return render_template('lostForm.html', message='Расширение файла должно быть: [png, jpg, jpeg]', title='добавление потеряшки')
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            db_sess = db_session.create_session()
            lost = Lost()
            if not(request.form['title']) or not(request.form['location']):
                return render_template('lostForm.html', message='Заполните все пропуски', title='добавление потеряшки')
            lost.title = request.form['title']
            lost.school = current_user.school
            lost.location = request.form['location']
            lost.imageLink = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            lost.userFound = current_user.id
            db_sess.add(lost)
            db_sess.commit()
            return redirect('/lostList')
    return render_template('lostForm.html', title='добавление потеряшки')


@app.route('/addEventForm', methods=['GET', 'POST'])  # страница для добавления события
def addEventForm():
    form = EventForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        event = Events()
        event.title = form.title.data
        event.description = form.description.data
        event.school = current_user.school
        event.userCreated = current_user.id
        event.classClub = form.classClub.data
        db_sess.add(event)
        db_sess.commit()
        return redirect('/eventList')
    return render_template('addEvent.html', title='Добавление события', form=form)


@app.route('/eventList', methods=['GET', 'POST'])  # список событий
def eventList():
    db_sess = db_session.create_session()
    events = db_sess.query(Events).filter(Events.school == current_user.school,
                                          Events.classClub == current_user.classClub).\
        order_by(Events.userCreated != current_user.id).all()
    return render_template('eventList.html', events=events, title='Список событий')


@app.route('/deleteEvent/<int:id>')  # удаление события
@login_required
def deleteEvent(id):
    db_sess = db_session.create_session()
    event = db_sess.query(Events).filter(Events.id == id).first()
    db_sess.delete(event)
    db_sess.commit()
    return redirect('/eventList')


def allowed_file(filename):  # проверка правильности расширения файла
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/deleteLost/<int:id>')  # удаление потерянной вещи
@login_required
def deleteLost(id):
    db_sess = db_session.create_session()
    lost = db_sess.query(Lost).filter(Lost.id == id).first()
    db_sess.delete(lost)
    db_sess.commit()
    os.remove(lost.imageLink)
    return redirect('/lostList')
'''

if __name__ == '__main__':
    db_session.global_init('db/plotter.db')
    app.run(debug=True)