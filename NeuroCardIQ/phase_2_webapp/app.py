from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, flash, Response, send_file)
from flask_login import login_required, current_user
from flask_bcrypt import Bcrypt
import os
import uuid
import traceback
import json
import io
import csv

from datetime import datetime
from dotenv import load_dotenv
from predictor import predict_risk
from admin import admin_bp
from database import db, User, Profile, Analysis
from auth     import auth, bcrypt, login_manager
load_dotenv()   # ← loads .env for local dev

# ─────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key')

# ── Database ──────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
DATABASE_URL = os.environ.get('DATABASE_URL',
               'sqlite:///' + os.path.join(BASE_DIR, 'mental_health.db'))

# Render gives postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI']        = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH']             = 16 * 1024 * 1024

# ── Init extensions ───────────────────────────────────────────
db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)

# ── Register blueprints ───────────────────────────────────────
app.register_blueprint(auth)

app.register_blueprint(admin_bp)

# ── Create tables ─────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("✅ Database tables created.")

# ── Upload folder ─────────────────────────────────────────────

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED

# ─────────────────────────────────────────────────────────────
# HELPER — build profile dict from current user
# ─────────────────────────────────────────────────────────────
def get_profile_dict(user):
    p = user.profile
    return {
        'name'             : p.name,
        'age'              : p.age,
        'gender'           : p.gender,
        'profession'       : p.profession,
        'avg_working_hours': p.avg_working_hours,
        'health_issues'    : p.health_issues    or '',
        'height_cm'        : p.height_cm        or 0,
        'weight_kg'        : p.weight_kg        or 0,
    }

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

# ── Landing ───────────────────────────────────────────────────
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main_profile'))
    return redirect(url_for('auth.signin'))

# ── Profile ───────────────────────────────────────────────────
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def main_profile():
    if request.method == 'POST':
        p = current_user.profile

        p.name              = request.form.get('name',
                                p.name).strip()
        p.age               = int(request.form.get('age', p.age))
        p.gender            = request.form.get('gender', p.gender)
        p.profession        = request.form.get('profession',
                                p.profession).strip()
        p.avg_working_hours = float(request.form.get(
                                'avg_working_hours',
                                p.avg_working_hours))
        p.health_issues     = request.form.get(
                                'health_issues', '').strip() or None

        # Height & weight (optional)
        try:
            hc = request.form.get('height_cm', '').strip()
            p.height_cm = float(hc) if hc else None
            if p.height_cm and not (100 <= p.height_cm <= 250):
                p.height_cm = None
        except ValueError:
            p.height_cm = None

        try:
            wk = request.form.get('weight_kg', '').strip()
            p.weight_kg = float(wk) if wk else None
            if p.weight_kg and not (20 <= p.weight_kg <= 300):
                p.weight_kg = None
        except ValueError:
            p.weight_kg = None

        p.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('main_profile'))

    # Stats for profile page
    analyses    = current_user.analyses
    total       = len(analyses)
    last        = analyses[0] if total > 0 else None
    risk_counts = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
    for a in analyses:
        if a.risk_name in risk_counts:
            risk_counts[a.risk_name] += 1

    # BMI for profile display
    p       = current_user.profile
    bmi_val = None
    bmi_cat = ''
    if p.height_cm and p.weight_kg and p.height_cm > 0:
        bmi_val = round(p.weight_kg / ((p.height_cm / 100) ** 2), 1)
        if   bmi_val < 16.0: 
            bmi_cat = 'Severely Underweight'
        elif bmi_val < 18.5: 
            bmi_cat = 'Underweight'
        elif bmi_val < 25.0: 
            bmi_cat = 'Normal Weight'
        elif bmi_val < 30.0: 
            bmi_cat = 'Overweight'
        elif bmi_val < 35.0: 
            
            bmi_cat = 'Obese Class I'
        elif bmi_val < 40.0: 
            bmi_cat = 'Obese Class II'
        else:                
            bmi_cat = 'Obese Class III'

    return render_template('profile.html',
                           user=current_user,
                           profile=p,
                           total_analyses=total,
                           last_analysis=last,
                           risk_counts=risk_counts,
                           bmi_val=bmi_val,
                           bmi_cat=bmi_cat)

# ── Analysis page ──────────────────────────────────────────────
@app.route('/analyze')
@login_required
def analyze():
    return render_template('index.html')

# ── Predict API ───────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # ── Validate uploads ──────────────────────────────────
        if 'eeg_image' not in request.files:
            return jsonify({'error': 'EEG image missing'}), 400
        if 'ecg_image' not in request.files:
            return jsonify({'error': 'ECG image missing'}), 400

        eeg_file = request.files['eeg_image']
        ecg_file = request.files['ecg_image']

        if not allowed_file(eeg_file.filename):
            return jsonify({'error': 'Invalid EEG file type'}), 400
        if not allowed_file(ecg_file.filename):
            return jsonify({'error': 'Invalid ECG file type'}), 400

        # ── Save temp files ───────────────────────────────────
        uid     = str(uuid.uuid4())[:8]
        eeg_ext = eeg_file.filename.rsplit('.', 1)[1].lower()
        ecg_ext = ecg_file.filename.rsplit('.', 1)[1].lower()

        eeg_path = os.path.join(UPLOAD_FOLDER, f"{uid}_eeg.{eeg_ext}")
        ecg_path = os.path.join(UPLOAD_FOLDER, f"{uid}_ecg.{ecg_ext}")

        eeg_file.save(eeg_path)
        ecg_file.save(ecg_path)

        # ── Run base prediction ───────────────────────────────
        result = predict_risk(eeg_path, ecg_path)

        # ── Personalise result ────────────────────────────────
        from personalizer import personalize
        profile_dict = {
            'name'             : current_user.profile.name,
            'age'              : current_user.profile.age,
            'gender'           : current_user.profile.gender,
            'profession'       : current_user.profile.profession,
            'avg_working_hours': current_user.profile.avg_working_hours,
            'health_issues'    : current_user.profile.health_issues or '',
        }
        past_analyses = current_user.analyses
        high_count    = sum(1 for a in past_analyses
                            if a.risk_level == 2)
        result = personalize(result, profile_dict,
                             high_count=high_count,
                             total=len(past_analyses))

        # ── Cleanup temp files ────────────────────────────────
        os.remove(eeg_path)
        os.remove(ecg_path)

        # ── Build full profile dict (height + weight included) ─
        profile_dict = get_profile_dict(current_user)

        # ── Personalise result ────────────────────────────────
        from personalizer import personalize
        past_analyses = current_user.analyses
        high_count    = sum(1 for a in past_analyses
                            if a.risk_level == 2)
        result = personalize(
            result,
            profile_dict,
            high_count = high_count,
            total      = len(past_analyses)
        )

        # ── Save analysis to DB ───────────────────────────────
        analysis = Analysis(
            user_id          = current_user.id,
            risk_level       = result['risk_level'],
            risk_name        = result['risk_name'],
            confidence       = result.get('adjusted_confidence',
                                          result['confidence']),
            prob_low         = result['prob_low'],
            prob_medium      = result['prob_medium'],
            prob_high        = result['prob_high'],
            brain_pct        = result['block_pct']['brain'],
            heart_pct        = result['block_pct']['heart'],
            interaction_pct  = result['block_pct']['interaction'],
            top_features_json= json.dumps(result['top_features']),
            eeg_summary_json = json.dumps(result['eeg_summary']),
            ecg_summary_json = json.dumps(result['ecg_summary']),
            xgb_prediction   = result['xgb_prediction'],
            rf_prediction    = result['rf_prediction'],
            models_agree     = result['models_agree'],
        )
        db.session.add(analysis)
        db.session.commit()

        result['analysis_id'] = analysis.id
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── Performance dashboard ──────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    analyses = current_user.analyses
    return render_template('dashboard.html',
                           user=current_user,
                           profile=current_user.profile,
                           analyses=[a.to_dict() for a in analyses])

# ── Export CSV ────────────────────────────────────────────────
@app.route('/export/csv')
@login_required
def export_csv():
    analyses = current_user.analyses
    output   = io.StringIO()
    writer   = csv.writer(output)

    writer.writerow([
        'Date', 'Risk Level', 'Confidence %',
        'Low %', 'Medium %', 'High %',
        'Brain %', 'Heart %', 'Interaction %',
        'XGB Prediction', 'RF Prediction', 'Models Agree'
    ])

    for a in analyses:
        writer.writerow([
            a.timestamp.strftime('%Y-%m-%d %H:%M'),
            a.risk_name,
            a.confidence,
            a.prob_low,
            a.prob_medium,
            a.prob_high,
            a.brain_pct,
            a.heart_pct,
            a.interaction_pct,
            a.xgb_prediction,
            a.rf_prediction,
            a.models_agree,
        ])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                f'attachment;filename=risk_history_{current_user.id}.csv'
        }
    )

# ── Export PDF ────────────────────────────────────────────────
@app.route('/export/pdf')
@login_required
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib           import colors
    from reportlab.platypus      import (SimpleDocTemplate, Table,
                                         TableStyle, Paragraph, Spacer)
    from reportlab.lib.styles    import getSampleStyleSheet

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems  = []
    p      = current_user.profile

    # ── Title ─────────────────────────────────────────────────
    elems.append(Paragraph(
        f"NeuroCardiQ Risk Report — {p.name}",
        styles['Title']))
    elems.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        styles['Normal']))
    elems.append(Spacer(1, 16))

    # ── Profile summary ───────────────────────────────────────
    elems.append(Paragraph("Personal Details", styles['Heading2']))

    # BMI calculation
    bmi_str = 'Not provided'
    if p.height_cm and p.weight_kg and p.height_cm > 0:
        bmi = round(p.weight_kg / ((p.height_cm / 100) ** 2), 1)
        if   bmi < 18.5: 
            cat = 'Underweight'
        elif bmi < 25.0: 
            cat = 'Normal'
        elif bmi < 30.0: 
            cat = 'Overweight'
        else:            
            cat = 'Obese'
        bmi_str = f"{bmi} ({cat})"

    profile_data = [
        ['Name',              p.name],
        ['Age',               str(p.age)],
        ['Gender',            p.gender],
        ['Profession',        p.profession],
        ['Avg Working Hours', f"{p.avg_working_hours}h/day"],
        ['Height',            f"{p.height_cm} cm" if p.height_cm else 'Not provided'],
        ['Weight',            f"{p.weight_kg} kg" if p.weight_kg else 'Not provided'],
        ['BMI',               bmi_str],
        ['Health Issues',     p.health_issues or 'None reported'],
    ]

    pt = Table(profile_data, colWidths=[160, 310])
    pt.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, -1), colors.HexColor('#1a6fbd')),
        ('TEXTCOLOR',     (0, 0), (0, -1), colors.white),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS',(1, 0), (-1, -1),
         [colors.HexColor('#f0f7ff'), colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING',       (0, 0), (-1, -1), 6),
    ]))
    elems.append(pt)
    elems.append(Spacer(1, 16))

    # ── Analysis history ──────────────────────────────────────
    analyses = current_user.analyses
    elems.append(Paragraph("Analysis History", styles['Heading2']))

    if analyses:
        headers = ['Date', 'Risk', 'Confidence',
                   'Low%', 'Med%', 'High%',
                   'Brain%', 'Heart%']
        rows = [headers]
        for a in analyses:
            rows.append([
                a.timestamp.strftime('%Y-%m-%d %H:%M'),
                a.risk_name,
                f"{a.confidence}%",
                f"{a.prob_low}%",
                f"{a.prob_medium}%",
                f"{a.prob_high}%",
                f"{a.brain_pct}%",
                f"{a.heart_pct}%",
            ])

        color_map = {
            'Low Risk'   : colors.HexColor('#d4f7e0'),
            'Medium Risk': colors.HexColor('#fff3cd'),
            'High Risk'  : colors.HexColor('#fde0e0'),
        }

        at = Table(rows, repeatRows=1)
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a6fbd')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('GRID',       (0, 0), (-1, -1), 0.3, colors.grey),
            ('PADDING',    (0, 0), (-1, -1), 5),
        ]
        for i, a in enumerate(analyses, 1):
            bg = color_map.get(a.risk_name, colors.white)
            style.append(('BACKGROUND', (0, i), (-1, i), bg))
        at.setStyle(TableStyle(style))
        elems.append(at)
    else:
        elems.append(Paragraph("No analyses yet.", styles['Normal']))

    doc.build(elems)
    buf.seek(0)
    return send_file(
        buf,
        mimetype='application/pdf',
        download_name=f'neurocardiq_report_{current_user.id}.pdf',
        as_attachment=True
    )

# ── Delete account ────────────────────────────────────────────
@app.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    from flask_login import logout_user
    from auth        import bcrypt as _bcrypt

    password = request.form.get('password', '')
    if _bcrypt.check_password_hash(
            current_user.password_hash, password):
        user = current_user._get_current_object()
        logout_user()
        db.session.delete(user)
        db.session.commit()
        flash('Your account has been permanently deleted.', 'info')
        return redirect(url_for('auth.signin'))

    flash('Incorrect password. Account not deleted.', 'danger')
    return redirect(url_for('main_profile'))

# ── Health check ──────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status'  : 'ok',
        'model'   : 'XGBoost',
        'accuracy': '97.58%',
        'app'     : 'NeuroCardiQ'
    })

# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Starting NeuroCardiQ Web App...")
    print("   Open http://localhost:5000")
    app.run(debug=True, port=5000)