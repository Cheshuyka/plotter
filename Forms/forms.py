from wtforms import IntegerField, StringField, BooleanField, SubmitField, EmailField, PasswordField, TextAreaField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired


class RegisterForm(FlaskForm):  # форма для регистрации
    name = StringField('Имя', validators=[DataRequired()])
    # organization = StringField('Организация', validators=[DataRequired()])
    email = EmailField('Почта', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    password_again = PasswordField('Повторите пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')


class LoginForm(FlaskForm):  # форма для входа в систему
    email = EmailField('Почта', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    remember_me = BooleanField('Запомнить меня')
    submit = SubmitField('Войти')