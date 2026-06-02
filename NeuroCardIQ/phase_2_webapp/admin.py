from flask          import (Blueprint, render_template, request,
                             redirect, url_for, flash, session,
                             Response)
from flask_login    import login_required, current_user
from functools      import wraps
from database       import db, User, Profile, Analysis
from auth           import bcrypt
from datetime       import datetime
import csv
import io

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# ─────────────────────────────────────────────────────────────
# ADMIN REQUIRED DECORATOR
# ─────────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in.', 'danger')
            return redirect(url_for('admin.admin_login'))
        if not current_user.is_admin:
            flash('Admin access required.', 'danger')
            return redirect(url_for('main_profile'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────
# ADMIN LOGIN
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for('admin.admin_dashboard'))

    error = None
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password   = request.form.get('password',   '').strip()

        user = (User.query.filter_by(email=identifier.lower()).first()
                or User.query.filter_by(phone=identifier).first())

        if (user and user.is_admin and
                bcrypt.check_password_hash(user.password_hash, password)):
            from flask_login import login_user
            login_user(user)
            flash(f'Welcome Admin {user.profile.name}!', 'success')
            return redirect(url_for('admin.admin_dashboard'))
        error = 'Invalid admin credentials.'

    return render_template('admin/login.html', error=error)


# ─────────────────────────────────────────────────────────────
# ADMIN DASHBOARD — system stats
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/dashboard')
@login_required
@admin_required
def admin_dashboard():
    total_users     = User.query.filter_by(is_admin=False).count()
    total_analyses  = Analysis.query.count()
    low_count       = Analysis.query.filter_by(risk_level=0).count()
    med_count       = Analysis.query.filter_by(risk_level=1).count()
    high_count      = Analysis.query.filter_by(risk_level=2).count()

    # Recent analyses (last 10)
    recent = (Analysis.query
              .order_by(Analysis.timestamp.desc())
              .limit(10).all())

    # Users with most analyses
    top_users = (db.session.query(User, db.func.count(Analysis.id)
                                  .label('cnt'))
                 .join(Analysis, User.id == Analysis.user_id)
                 .filter(User.is_admin == False)
                 .group_by(User.id)
                 .order_by(db.text('cnt DESC'))
                 .limit(5).all())

    # Gender distribution
    gender_dist = {}
    for p in Profile.query.all():
        g = p.gender or 'Unknown'
        gender_dist[g] = gender_dist.get(g, 0) + 1

    # Risk by profession tier
    from personalizer import get_profession_tier
    prof_risk = {'high': [0,0,0], 'medium': [0,0,0], 'low': [0,0,0]}
    for a in Analysis.query.all():
        u = User.query.get(a.user_id)
        if u and u.profile:
            tier = get_profession_tier(u.profile.profession)
            prof_risk[tier][a.risk_level] += 1

    # Average confidence
    avg_conf = (db.session.query(db.func.avg(Analysis.confidence))
                .scalar() or 0)

    # New users last 7 days
    from datetime import timedelta
    week_ago   = datetime.utcnow() - timedelta(days=7)
    new_users  = User.query.filter(
                     User.created_at >= week_ago,
                     User.is_admin == False).count()

    return render_template('admin/dashboard.html',
        total_users    = total_users,
        total_analyses = total_analyses,
        low_count      = low_count,
        med_count      = med_count,
        high_count     = high_count,
        recent         = recent,
        top_users      = top_users,
        gender_dist    = gender_dist,
        prof_risk      = prof_risk,
        avg_conf       = round(avg_conf, 1),
        new_users      = new_users,
    )


# ─────────────────────────────────────────────────────────────
# ALL USERS LIST
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/users')
@login_required
@admin_required
def admin_users():
    search = request.args.get('q', '').strip()
    risk_f = request.args.get('risk', '').strip()
    sort   = request.args.get('sort', 'newest')

    query = User.query.filter_by(is_admin=False)

    # Search by name/email/phone/profession
    if search:
        query = query.join(Profile).filter(
            db.or_(
                User.email.ilike(f'%{search}%'),
                User.phone.ilike(f'%{search}%'),
                Profile.name.ilike(f'%{search}%'),
                Profile.profession.ilike(f'%{search}%'),
            )
        )

    # Sort
    if sort == 'oldest':
        query = query.order_by(User.created_at.asc())
    elif sort == 'name':
        query = query.join(Profile, isouter=True)\
                     .order_by(Profile.name.asc())
    else:
        query = query.order_by(User.created_at.desc())

    users = query.all()

    # Filter by latest risk level
    if risk_f in ('0', '1', '2'):
        filtered = []
        for u in users:
            if u.analyses:
                latest = u.analyses[0]
                if str(latest.risk_level) == risk_f:
                    filtered.append(u)
        users = filtered

    # Attach stats to each user
    user_stats = []
    for u in users:
        analyses   = u.analyses
        total      = len(analyses)
        latest     = analyses[0] if total > 0 else None
        high_cnt   = sum(1 for a in analyses if a.risk_level == 2)
        avg_conf   = (sum(a.confidence for a in analyses) / total
                      if total > 0 else 0)

        bmi = None
        if u.profile and u.profile.height_cm and u.profile.weight_kg:
            bmi = round(u.profile.weight_kg /
                        ((u.profile.height_cm / 100) ** 2), 1)

        user_stats.append({
            'user'     : u,
            'profile'  : u.profile,
            'total'    : total,
            'latest'   : latest,
            'high_cnt' : high_cnt,
            'avg_conf' : round(avg_conf, 1),
            'bmi'      : bmi,
        })

    return render_template('admin/users.html',
                           user_stats=user_stats,
                           search=search,
                           risk_f=risk_f,
                           sort=sort,
                           total=len(user_stats))


# ─────────────────────────────────────────────────────────────
# SINGLE USER DETAIL
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/user/<int:user_id>')
@login_required
@admin_required
def admin_user_detail(user_id):
    user     = User.query.get_or_404(user_id)
    analyses = user.analyses
    profile  = user.profile

    bmi = bmi_cat = None
    if profile and profile.height_cm and profile.weight_kg:
        bmi = round(profile.weight_kg /
                    ((profile.height_cm / 100) ** 2), 1)
        if   bmi < 18.5: 
            bmi_cat = 'Underweight'
        elif bmi < 25.0: 
            bmi_cat = 'Normal'
        elif bmi < 30.0: 
            bmi_cat = 'Overweight'
        else:            
            bmi_cat = 'Obese'

    total     = len(analyses)
    low_cnt   = sum(1 for a in analyses if a.risk_level == 0)
    med_cnt   = sum(1 for a in analyses if a.risk_level == 1)
    high_cnt  = sum(1 for a in analyses if a.risk_level == 2)
    avg_conf  = round(sum(a.confidence for a in analyses) / total, 1) \
                if total > 0 else 0

    # Trend
    trend = 'No data'
    if total >= 2:
        if analyses[0].risk_level < analyses[1].risk_level:
            trend = '📉 Improving'
        elif analyses[0].risk_level > analyses[1].risk_level:
            trend = '📈 Worsening'
        else:
            trend = '➡️ Stable'

    return render_template('admin/user_detail.html',
        user      = user,
        profile   = profile,
        analyses  = analyses,
        bmi       = bmi,
        bmi_cat   = bmi_cat,
        total     = total,
        low_cnt   = low_cnt,
        med_cnt   = med_cnt,
        high_cnt  = high_cnt,
        avg_conf  = avg_conf,
        trend     = trend,
    )


# ─────────────────────────────────────────────────────────────
# EXPORT USER DATA — CSV
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/export/<int:user_id>')
@login_required
@admin_required
def admin_export_user(user_id):
    user     = User.query.get_or_404(user_id)
    analyses = user.analyses
    profile  = user.profile

    output = io.StringIO()
    writer = csv.writer(output)

    # Profile info header
    writer.writerow(['=== USER PROFILE ==='])
    writer.writerow(['Name',       profile.name       if profile else ''])
    writer.writerow(['Email',      user.email         or ''])
    writer.writerow(['Phone',      user.phone         or ''])
    writer.writerow(['Age',        profile.age        if profile else ''])
    writer.writerow(['Gender',     profile.gender     if profile else ''])
    writer.writerow(['Profession', profile.profession if profile else ''])
    writer.writerow(['Avg Hours',  f"{profile.avg_working_hours}h/day"
                                   if profile else ''])
    writer.writerow(['Height',     f"{profile.height_cm}cm"
                                   if profile and profile.height_cm else ''])
    writer.writerow(['Weight',     f"{profile.weight_kg}kg"
                                   if profile and profile.weight_kg else ''])
    writer.writerow(['Health',     profile.health_issues
                                   if profile else ''])
    writer.writerow(['Joined',     user.created_at.strftime('%Y-%m-%d')])
    writer.writerow([])

    # Analysis data
    writer.writerow(['=== ANALYSES ==='])
    writer.writerow(['#', 'Date', 'Risk', 'Confidence',
                     'Low%', 'Med%', 'High%',
                     'Brain%', 'Heart%', 'Interaction%',
                     'XGB', 'RF', 'Agree'])
    for i, a in enumerate(analyses, 1):
        writer.writerow([
            i,
            a.timestamp.strftime('%Y-%m-%d %H:%M'),
            a.risk_name,
            f"{a.confidence}%",
            f"{a.prob_low}%",
            f"{a.prob_medium}%",
            f"{a.prob_high}%",
            f"{a.brain_pct}%",
            f"{a.heart_pct}%",
            f"{a.interaction_pct}%",
            a.xgb_prediction,
            a.rf_prediction,
            'Yes' if a.models_agree else 'No',
        ])

    output.seek(0)
    name = profile.name.replace(' ', '_') if profile else str(user_id)
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                f'attachment;filename=user_{name}_data.csv'
        }
    )


# ─────────────────────────────────────────────────────────────
# DELETE USER
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash('Cannot delete admin accounts.', 'danger')
        return redirect(url_for('admin.admin_users'))

    name = user.profile.name if user.profile else str(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{name}" deleted successfully.', 'success')
    return redirect(url_for('admin.admin_users'))


# ─────────────────────────────────────────────────────────────
# CREATE ADMIN — one time setup route
# Visit /admin/setup to create the first admin account
# DISABLE THIS AFTER FIRST USE by commenting it out
# ─────────────────────────────────────────────────────────────
@admin_bp.route('/setup', methods=['GET', 'POST'])
def admin_setup():
    # Block if admin already exists
    if User.query.filter_by(is_admin=True).first():
        flash('Admin already exists. Setup disabled.', 'danger')
        return redirect(url_for('admin.admin_login'))

    error = None
    if request.method == 'POST':
        email    = request.form.get('email',    '').strip().lower()
        password = request.form.get('password', '').strip()
        name     = request.form.get('name',     '').strip()

        if not email or not password or not name:
            error = 'All fields required.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters.'
        elif User.query.filter_by(email=email).first():
            error = 'Email already in use.'
        else:
            pw_hash = bcrypt.generate_password_hash(password)\
                            .decode('utf-8')
            admin_user = User(
                email         = email,
                password_hash = pw_hash,
                is_admin      = True,
            )
            db.session.add(admin_user)
            db.session.flush()
            db.session.add(Profile(
                user_id           = admin_user.id,
                name              = name,
                age               = 30,
                gender            = 'Prefer not to say',
                profession        = 'System Administrator',
                avg_working_hours = 8,
            ))
            db.session.commit()
            flash('Admin account created! Please log in.', 'success')
            return redirect(url_for('admin.admin_login'))

    return render_template('admin/setup.html', error=error)