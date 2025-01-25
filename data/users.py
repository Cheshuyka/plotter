import sqlalchemy
from .db_session import SqlAlchemyBase
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy_serializer import SerializerMixin
from sqlalchemy import orm


class User(SqlAlchemyBase, UserMixin, SerializerMixin):  # пользователи
    __tablename__ = 'users'
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    name = sqlalchemy.Column(sqlalchemy.String)
    organization = sqlalchemy.Column(sqlalchemy.String)
    # position = sqlalchemy.Column(sqlalchemy.String, default='Не выбрано')
    email = sqlalchemy.Column(sqlalchemy.String, unique=True)
    dataset_name = sqlalchemy.Column(sqlalchemy.String, unique=True)
    hashed_password = sqlalchemy.Column(sqlalchemy.String)

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)