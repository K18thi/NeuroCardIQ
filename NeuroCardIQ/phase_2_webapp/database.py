from flask_sqlalchemy import SQLAlchemy
from flask_login      import UserMixin
from datetime         import datetime

db = SQLAlchemy()

# ─────────────────────────────────────────────────────────────
# USER TABLE
# ─────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer,     primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=True)
    phone         = db.Column(db.String(20),  unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active     = db.Column(db.Boolean,     default=True)
    is_admin      = db.Column(db.Boolean,     default=False)  # ← ADDED

    profile  = db.relationship('Profile',  backref='user',
                                uselist=False,
                                cascade='all, delete-orphan')
    analyses = db.relationship('Analysis', backref='user',
                                cascade='all, delete-orphan',
                                order_by='Analysis.timestamp.desc()')

    def __repr__(self):
        return f'<User {self.email or self.phone}>'


# ─────────────────────────────────────────────────────────────
# PROFILE TABLE
# ─────────────────────────────────────────────────────────────
class Profile(db.Model):
    __tablename__ = 'profiles'

    id                = db.Column(db.Integer,     primary_key=True)
    user_id           = db.Column(db.Integer,
                                  db.ForeignKey('users.id'), nullable=False)
    name              = db.Column(db.String(100), nullable=False)
    age               = db.Column(db.Integer,     nullable=False)
    gender            = db.Column(db.String(20),  nullable=False)
    profession        = db.Column(db.String(100), nullable=False)
    avg_working_hours = db.Column(db.Float,       nullable=False)
    health_issues     = db.Column(db.Text,        nullable=True)
    height_cm         = db.Column(db.Float,       nullable=True)
    weight_kg         = db.Column(db.Float,       nullable=True)
    updated_at        = db.Column(db.DateTime,    default=datetime.utcnow,
                                  onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'name'             : self.name,
            'age'              : self.age,
            'gender'           : self.gender,
            'profession'       : self.profession,
            'avg_working_hours': self.avg_working_hours,
            'health_issues'    : self.health_issues or '',
            'height_cm'        : self.height_cm,
            'weight_kg'        : self.weight_kg,
        }


# ─────────────────────────────────────────────────────────────
# ANALYSIS TABLE
# ─────────────────────────────────────────────────────────────
class Analysis(db.Model):
    __tablename__ = 'analyses'

    id                = db.Column(db.Integer,     primary_key=True)
    user_id           = db.Column(db.Integer,
                                  db.ForeignKey('users.id'), nullable=False)
    timestamp         = db.Column(db.DateTime,    default=datetime.utcnow)

    risk_level        = db.Column(db.Integer,     nullable=False)
    risk_name         = db.Column(db.String(20),  nullable=False)
    confidence        = db.Column(db.Float,       nullable=False)
    prob_low          = db.Column(db.Float,       nullable=False)
    prob_medium       = db.Column(db.Float,       nullable=False)
    prob_high         = db.Column(db.Float,       nullable=False)

    brain_pct         = db.Column(db.Float,       nullable=False)
    heart_pct         = db.Column(db.Float,       nullable=False)
    interaction_pct   = db.Column(db.Float,       nullable=False)

    top_features_json = db.Column(db.Text,        nullable=True)
    eeg_summary_json  = db.Column(db.Text,        nullable=True)
    ecg_summary_json  = db.Column(db.Text,        nullable=True)

    xgb_prediction    = db.Column(db.String(20),  nullable=True)
    rf_prediction     = db.Column(db.String(20),  nullable=True)
    models_agree      = db.Column(db.Boolean,     nullable=True)

    def to_dict(self):
        import json
        return {
            'id'             : self.id,
            'timestamp'      : self.timestamp.strftime('%Y-%m-%d %H:%M'),
            'risk_level'     : self.risk_level,
            'risk_name'      : self.risk_name,
            'confidence'     : self.confidence,
            'prob_low'       : self.prob_low,
            'prob_medium'    : self.prob_medium,
            'prob_high'      : self.prob_high,
            'brain_pct'      : self.brain_pct,
            'heart_pct'      : self.heart_pct,
            'interaction_pct': self.interaction_pct,
            'top_features'   : json.loads(self.top_features_json or '[]'),
            'eeg_summary'    : json.loads(self.eeg_summary_json  or '{}'),
            'ecg_summary'    : json.loads(self.ecg_summary_json  or '{}'),
            'xgb_prediction' : self.xgb_prediction,
            'rf_prediction'  : self.rf_prediction,
            'models_agree'   : self.models_agree,
        }