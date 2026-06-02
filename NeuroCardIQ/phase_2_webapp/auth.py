from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import db, User, Profile
import re

bcrypt = Bcrypt()
login_manager = LoginManager()

login_manager.login_view = 'auth.signin'
login_manager.login_message = 'Please sign in to continue.'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


auth = Blueprint('auth', __name__)


def is_valid_email(e):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', e)


def is_valid_phone(p):
    return re.match(r'^\+?[\d\s\-]{7,15}$', p)


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main_profile'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm', '').strip()
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        profession = request.form.get('profession', '').strip()
        avg_hours = request.form.get('avg_working_hours', '').strip()
        health = request.form.get('health_issues', '').strip()
        height_cm = request.form.get('height_cm', '').strip()
        weight_kg = request.form.get('weight_kg', '').strip()

        errors = []
        email = None
        phone = None

        # Email / Phone validation
        if is_valid_email(identifier):
            email = identifier.lower()
            if User.query.filter_by(email=email).first():
                errors.append('Email already registered.')
        elif is_valid_phone(identifier):
            phone = identifier
            if User.query.filter_by(phone=phone).first():
                errors.append('Phone already registered.')
        else:
            errors.append('Enter a valid email or phone number.')

        # Password validation
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')

        # Basic validations
        if not name:
            errors.append('Name is required.')

        try:
            age = int(age)
            if not (10 <= age <= 120):
                raise ValueError
        except ValueError:
            errors.append('Enter a valid age (10–120).')

        if not gender:
            errors.append('Gender is required.')

        if not profession:
            errors.append('Profession is required.')

        try:
            avg_hours = float(avg_hours)
            if not (0 <= avg_hours <= 24):
                raise ValueError
        except ValueError:
            errors.append('Enter valid working hours (0–24).')

        # Height validation
        try:
            height_cm = float(height_cm) if height_cm else None
            if height_cm and not (100 <= height_cm <= 250):
                height_cm = None
        except ValueError:
            height_cm = None

        # Weight validation
        try:
            weight_kg = float(weight_kg) if weight_kg else None
            if weight_kg and not (30 <= weight_kg <= 300):
                weight_kg = None
        except ValueError:
            weight_kg = None

        # If errors → return
        if errors:
            return render_template('signup.html', errors=errors, form=request.form)

        # Create user
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, phone=phone, password_hash=pw_hash)

        db.session.add(user)
        db.session.flush()

        # Create profile (ONLY ONCE ✅)
        profile = Profile(
            user_id=user.id,
            name=name,
            age=age,
            gender=gender,
            profession=profession,
            avg_working_hours=avg_hours,
            health_issues=health or None,
            height_cm=height_cm,
            weight_kg=weight_kg
        )

        db.session.add(profile)
        db.session.commit()

        login_user(user)
        flash('Account created! Welcome aboard.', 'success')
        return redirect(url_for('main_profile'))

    return render_template('signup.html', errors=[], form={})


@auth.route('/signin', methods=['GET', 'POST'])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('main_profile'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '').strip()
        remember = request.form.get('remember') == 'on'

        user = None

        if is_valid_email(identifier):
            user = User.query.filter_by(email=identifier.lower()).first()
        elif is_valid_phone(identifier):
            user = User.query.filter_by(phone=identifier).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.profile.name}!', 'success')
            return redirect(url_for('main_profile'))

        return render_template('signin.html', error='Invalid credentials. Try again.')

    return render_template('signin.html', error=None)


@auth.route('/signout')
@login_required
def signout():
    logout_user()
    flash('You have been signed out.', 'info')
    return redirect(url_for('auth.signin'))