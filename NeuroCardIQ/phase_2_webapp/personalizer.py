import numpy as np

# ─────────────────────────────────────────────────────────────
# BMI CALCULATOR
# ─────────────────────────────────────────────────────────────
def calculate_bmi(height_cm, weight_kg):
    """Returns (bmi_value, bmi_category, bmi_risk_tier)"""
    if not height_cm or not weight_kg or height_cm < 100:
        return None, 'Unknown', 'neutral'
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if   bmi < 16.0: 
        return round(bmi,1), 'Severely Underweight', 'high'
    elif bmi < 18.5: 
        return round(bmi,1), 'Underweight',          'medium'
    elif bmi < 25.0: 
        return round(bmi,1), 'Normal Weight',        'low'
    elif bmi < 30.0: 
        return round(bmi,1), 'Overweight',           'medium'
    elif bmi < 35.0: 
        return round(bmi,1), 'Obese Class I',        'high'
    elif bmi < 40.0: 
        return round(bmi,1), 'Obese Class II',       'high'
    else:            
        return round(bmi,1), 'Obese Class III',       'high'


# ─────────────────────────────────────────────────────────────
# PROFESSION RISK TIERS
# ─────────────────────────────────────────────────────────────
PROFESSION_RISK = {
    'high': [
        'doctor','physician','surgeon','nurse','paramedic',
        'emergency','icu','healthcare','medical',
        'police','firefighter','military','soldier',
        'lawyer','attorney','judge','advocate',
        'investment banker','trader','stockbroker',
        'journalist','reporter','news anchor',
        'social worker','therapist','counselor',
        'pilot','air traffic controller',
    ],
    'medium': [
        'engineer','developer','programmer','software',
        'manager','team lead','project manager','scrum',
        'teacher','professor','lecturer','educator',
        'accountant','auditor','finance','analyst',
        'architect','designer','researcher','scientist',
        'consultant','advisor','strategist',
        'entrepreneur','founder','startup',
        'sales','marketing','business development',
    ],
    'low': [
        'student','intern','trainee',
        'artist','musician','writer','author','poet',
        'farmer','gardener','nature',
        'retired','homemaker','housewife','househusband',
        'librarian','archivist',
        'yoga','meditation','wellness',
    ],
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_age_group(age):
    if age < 18: return 'teen'
    if age < 25: return 'young_adult'
    if age < 35: return 'adult'
    if age < 50: return 'middle_age'
    if age < 65: return 'senior'
    return 'elderly'

AGE_GROUP_LABELS = {
    'teen'        : 'Teenager (under 18)',
    'young_adult' : 'Young Adult (18–24)',
    'adult'       : 'Adult (25–34)',
    'middle_age'  : 'Middle-Aged (35–49)',
    'senior'      : 'Senior (50–64)',
    'elderly'     : 'Elderly (65+)',
}

def get_profession_tier(profession):
    prof_lower = profession.lower()
    for tier, keywords in PROFESSION_RISK.items():
        if any(kw in prof_lower for kw in keywords):
            return tier
    return 'medium'

def get_hours_category(hours):
    if hours <= 6:  return 'underworked'
    if hours <= 8:  return 'healthy'
    if hours <= 10: return 'overworked'
    if hours <= 12: return 'excessive'
    return 'extreme'

HOURS_LABELS = {
    'underworked': 'Low (<6h)',
    'healthy'    : 'Healthy (6–8h)',
    'overworked' : 'Overworked (8–10h)',
    'excessive'  : 'Excessive (10–12h)',
    'extreme'    : 'Extreme (12h+)',
}


# ─────────────────────────────────────────────────────────────
# SOFT LABEL NUDGE — defined first so personalize() can call it
# ─────────────────────────────────────────────────────────────
def soft_label_nudge(prob_low, prob_medium, prob_high, profile):
    """
    Softly adjusts risk probabilities based on profile.
    Only activates when model is uncertain (close probabilities).
    Never overrides a confident model prediction.
    """
    age    = profile.get('age', 30)
    hours  = profile.get('avg_working_hours', 8)
    prof   = profile.get('profession', '').lower()
    health = (profile.get('health_issues') or '').lower()

    # Build profile risk weight
    weight = 0.0
    if hours > 12:       
        weight += 0.15
    elif hours > 10:     
        weight += 0.10
    elif hours > 8:      
        weight += 0.05

    if age > 50:         
        weight += 0.10
    elif age > 40:       
        weight += 0.05
    elif age < 18:       
        weight += 0.05

    prof_tier = get_profession_tier(prof)
    if prof_tier == 'high':  
        weight += 0.12
    elif prof_tier == 'low': 
        weight -= 0.05

    high_risk_conditions = [
        'anxiety','depression','ptsd','hypertension',
        'heart','bipolar','insomnia','panic'
    ]
    matched = [c for c in high_risk_conditions if c in health]
    weight += min(len(matched) * 0.05, 0.15)

    # Check if model is uncertain
    probs    = [prob_low, prob_medium, prob_high]
    sorted_p = sorted(probs, reverse=True)
    uncertain = (sorted_p[0] - sorted_p[1]) < 0.12

    if uncertain and weight > 0.15:
        shift       = weight * 0.25
        prob_high   = min(0.95, prob_high + shift)
        prob_low    = max(0.01, prob_low  - shift)
        total       = prob_low + prob_medium + prob_high
        prob_low   /= total
        prob_medium /= total
        prob_high  /= total
        new_probs   = [prob_low, prob_medium, prob_high]
        risk_level  = int(np.argmax(new_probs))
        nudged      = True
    else:
        risk_level  = int(np.argmax(probs))
        nudged      = False

    return {
        'prob_low'    : round(prob_low    * 100, 1),
        'prob_medium' : round(prob_medium * 100, 1),
        'prob_high'   : round(prob_high   * 100, 1),
        'risk_level'  : risk_level,
        'risk_name'   : ['Low Risk','Medium Risk','High Risk'][risk_level],
        'nudged'      : nudged,
        'nudge_weight': round(weight, 3),
    }


# ─────────────────────────────────────────────────────────────
# CONFIDENCE ADJUSTMENT
# ─────────────────────────────────────────────────────────────
def adjust_confidence(base_confidence, risk_level, profile):
    """
    Moderately nudge confidence based on profile risk factors.
    Never changes the risk LABEL — only confidence score.
    Max adjustment: ±15 percentage points.
    """
    adjustment = 0.0
    factors    = []

    age        = profile.get('age', 30)
    hours      = profile.get('avg_working_hours', 8)
    profession = profile.get('profession', '')
    health     = profile.get('health_issues', '') or ''
    gender     = profile.get('gender', '')

    prof_tier  = get_profession_tier(profession)
    age_group  = get_age_group(age)
    hours_cat  = get_hours_category(hours)

    # ── Age factor ────────────────────────────────────────────
    if age_group in ('senior', 'elderly') and risk_level >= 1:
        adjustment += 4.0
        factors.append(f'Age ({age}) increases vulnerability')
    elif age_group == 'teen' and risk_level >= 1:
        adjustment += 3.0
        factors.append(f'Young age ({age}) — developing stress response')
    elif age_group == 'young_adult' and risk_level == 0:
        adjustment -= 2.0
        factors.append(f'Age ({age}) — resilient age group')

    # ── Gender factor ─────────────────────────────────────────
    if gender.lower() == 'female' and risk_level >= 1:
        adjustment += 2.5
        factors.append('Females show higher HRV sensitivity to stress')
    elif gender.lower() == 'male' and risk_level == 2:
        adjustment += 2.0
        factors.append('Males tend to underreport stress symptoms')

    # ── Profession factor ─────────────────────────────────────
    if prof_tier == 'high' and risk_level >= 1:
        adjustment += 5.0
        factors.append(f'High-stress profession ({profession})')
    elif prof_tier == 'high' and risk_level == 0:
        adjustment += 2.0
        factors.append('High-stress profession — monitor closely')
    elif prof_tier == 'low' and risk_level == 0:
        adjustment -= 3.0
        factors.append(f'Low-stress profession ({profession})')

    # ── Working hours factor ──────────────────────────────────
    if hours_cat == 'extreme' and risk_level >= 1:
        adjustment += 5.0
        factors.append(f'Extreme working hours ({hours}h/day)')
    elif hours_cat == 'excessive' and risk_level >= 1:
        adjustment += 3.5
        factors.append(f'Excessive working hours ({hours}h/day)')
    elif hours_cat == 'overworked' and risk_level >= 1:
        adjustment += 2.0
        factors.append(f'Above-recommended working hours ({hours}h/day)')
    elif hours_cat == 'healthy' and risk_level == 0:
        adjustment -= 2.0
        factors.append(f'Healthy working hours ({hours}h/day)')

    # ── Health issues factor ──────────────────────────────────
    high_risk_conditions = [
        'anxiety','depression','ptsd','bipolar','schizophrenia',
        'hypertension','heart','cardiac','diabetes','thyroid',
        'insomnia','sleep','panic','ocd','adhd','burnout',
    ]
    if health:
        matched = [c for c in high_risk_conditions
                   if c in health.lower()]
        if matched and risk_level >= 1:
            adjustment += min(len(matched) * 2.5, 7.0)
            factors.append(
                f'Pre-existing conditions: {", ".join(matched[:3])}')
        elif matched and risk_level == 0:
            adjustment += min(len(matched) * 1.5, 4.0)
            factors.append(
                f'Health history warrants monitoring: '
                f'{", ".join(matched[:2])}')

    # ── BMI factor ────────────────────────────────────────────
    height  = profile.get('height_cm', 0) or 0
    weight  = profile.get('weight_kg', 0) or 0
    bmi_val, bmi_cat, bmi_tier = calculate_bmi(height, weight)

    if bmi_val:
        if bmi_tier == 'high' and risk_level >= 1:
            adjustment += 4.0
            factors.append(
                f'BMI {bmi_val} ({bmi_cat}) — elevated metabolic risk')
        elif bmi_tier == 'high' and risk_level == 0:
            adjustment += 2.0
            factors.append(
                f'BMI {bmi_val} ({bmi_cat}) — monitor mental-physical link')
        elif bmi_tier == 'medium' and risk_level >= 1:
            adjustment += 2.0
            factors.append(
                f'BMI {bmi_val} ({bmi_cat}) — moderate metabolic factor')
        elif bmi_tier == 'low' and risk_level == 0:
            adjustment -= 1.5
            factors.append(
                f'BMI {bmi_val} ({bmi_cat}) — healthy weight is protective')

    # ── Clamp and return ──────────────────────────────────────
    adjustment = max(-15.0, min(15.0, adjustment))
    adjusted   = round(max(5.0, min(99.0,
                    base_confidence + adjustment)), 1)

    return adjusted, factors


# ─────────────────────────────────────────────────────────────
# RISK INTERPRETATION
# ─────────────────────────────────────────────────────────────
def get_risk_interpretation(risk_level, risk_name, profile,
                             confidence, personal_factors):
    age        = profile.get('age', 30)
    gender     = profile.get('gender', 'Unknown')
    profession = profile.get('profession', 'your profession')
    hours      = profile.get('avg_working_hours', 8)
    health     = profile.get('health_issues', '') or ''

    age_group  = get_age_group(age)
    prof_tier  = get_profession_tier(profession)
    hours_cat  = get_hours_category(hours)

    # ── Base interpretation ───────────────────────────────────
    if risk_level == 0:
        base = (
            "Your brain–heart signals indicate LOW mental health risk. "
            "Your EEG and ECG patterns suggest good neurological and "
            "cardiovascular regulation at this time."
        )
    elif risk_level == 1:
        base = (
            "Your brain–heart signals indicate MODERATE mental health risk. "
            "Your EEG and ECG patterns show some stress-related changes "
            "that warrant attention and self-care."
        )
    else:
        base = (
            "Your brain–heart signals indicate HIGH mental health risk. "
            "Your EEG and ECG patterns show significant stress markers "
            "that require prompt attention."
        )

    # ── Age context ───────────────────────────────────────────
    age_context = {
        'teen'       : (f"At {age}, your brain is still developing — "
                        f"stress at this stage can have lasting effects."),
        'young_adult': ("In your mid-20s, establishing healthy stress "
                        "habits now creates lifelong resilience."),
        'adult'      : (f"At {age}, career and life demands often peak — "
                        f"proactive mental health care is essential."),
        'middle_age' : ("In your 40s, cumulative stress can compound "
                        "quickly — regular monitoring is recommended."),
        'senior'     : (f"At {age}, the body's stress recovery slows — "
                        f"consistent self-care routines are vital."),
        'elderly'    : (f"At {age}, mental and physical health are closely "
                        f"linked — any risk signals deserve clinical review."),
    }.get(age_group, '')

    # ── Gender context ────────────────────────────────────────
    gender_context = ''
    if gender.lower() == 'female':
        gender_context = (
            "Women often experience stress differently — hormonal cycles "
            "can influence HRV and EEG patterns."
        )
    elif gender.lower() == 'male':
        gender_context = (
            "Men often mask stress symptoms — physical biomarkers like "
            "HRV can reveal hidden stress before it becomes critical."
        )

    # ── Profession context ────────────────────────────────────
    if prof_tier == 'high':
        prof_context = (
            f"As a {profession}, you operate in a high-demand environment. "
            f"Occupational stress is a significant risk factor for your role."
        )
    elif prof_tier == 'medium':
        prof_context = (
            f"Your role as a {profession} carries moderate stress exposure. "
            f"Work-life boundaries are key to maintaining mental wellness."
        )
    else:
        prof_context = (
            f"Your profession ({profession}) is generally lower stress, "
            f"which is a protective factor for mental health."
        )

    # ── Working hours context ─────────────────────────────────
    hours_context = {
        'underworked': (f"Working {hours}h/day — ensure you stay socially "
                        f"and mentally engaged to avoid isolation."),
        'healthy'    : (f"Your {hours}h/day workload is within healthy "
                        f"limits — maintain this balance."),
        'overworked' : (f"At {hours}h/day you're exceeding recommended "
                        f"limits — fatigue accumulation is a real risk."),
        'excessive'  : (f"Working {hours}h/day is excessive. Chronic "
                        f"overwork significantly elevates mental health risk."),
        'extreme'    : (f"At {hours}h/day, you are in the extreme overwork "
                        f"zone. This is a critical risk factor."),
    }.get(hours_cat, '')

    # ── Health issues context ─────────────────────────────────
    health_context = ''
    if health:
        health_context = (
            f"Your reported health conditions ({health}) can interact "
            f"with mental health risk — integrated care is recommended."
        )

    # ── BMI context ───────────────────────────────────────────
    height  = profile.get('height_cm', 0) or 0
    weight  = profile.get('weight_kg', 0) or 0
    bmi_val, bmi_cat, bmi_tier = calculate_bmi(height, weight)

    bmi_context = ''
    if bmi_val:
        if bmi_tier == 'high':
            bmi_context = (
                f"Your BMI of {bmi_val} ({bmi_cat}) is clinically significant. "
                f"Obesity and severe underweight both independently increase "
                f"cortisol levels and worsen HRV — directly affecting your "
                f"brain-heart risk score."
            )
        elif bmi_tier == 'medium':
            bmi_context = (
                f"Your BMI of {bmi_val} ({bmi_cat}) is slightly outside "
                f"the optimal range. Even modest weight normalisation "
                f"improves HRV and reduces mental health risk markers."
            )
        else:
            bmi_context = (
                f"Your BMI of {bmi_val} ({bmi_cat}) is within the healthy "
                f"range — this is a strong protective factor for both "
                f"cardiovascular and mental health."
            )

    return {
        'summary'        : base,
        'age_context'    : age_context,
        'gender_context' : gender_context,
        'prof_context'   : prof_context,
        'hours_context'  : hours_context,
        'health_context' : health_context,
        'bmi_context'    : bmi_context,
    }


# ─────────────────────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────────────────────
def get_recommendations(risk_level, profile,
                         high_count=0, total=0):
    age        = profile.get('age', 30)
    gender     = profile.get('gender', '')
    profession = profile.get('profession', '')
    hours      = profile.get('avg_working_hours', 8)
    health     = profile.get('health_issues', '') or ''

    age_group  = get_age_group(age)
    prof_tier  = get_profession_tier(profession)
    hours_cat  = get_hours_category(hours)
    recs       = []

    # ── 1. Risk-level base ────────────────────────────────────
    if risk_level == 2:
        recs.append({
            'icon'    : '🏥',
            'title'   : 'Seek Professional Support',
            'desc'    : (
                f"High risk detected. Consider speaking with a "
                f"mental health professional soon. "
                f"{'For healthcare workers like you, peer support programs exist.' if prof_tier == 'high' else 'Early intervention leads to much better outcomes.'}"),
            'priority': 'high'
        })
    elif risk_level == 1:
        recs.append({
            'icon'    : '⚠️',
            'title'   : 'Monitor Your Stress Closely',
            'desc'    : (
                "Medium risk detected. Track how your risk evolves "
                "over the next 2–4 weeks with regular scans."),
            'priority': 'medium'
        })
    else:
        recs.append({
            'icon'    : '✅',
            'title'   : 'Maintain Your Healthy State',
            'desc'    : (
                "Low risk — great work! Keep up your current "
                "habits and schedule monthly check-ins."),
            'priority': 'low'
        })

    # ── 2. Profession-specific ────────────────────────────────
    prof_lower = profession.lower()
    if any(k in prof_lower for k in
           ['doctor','nurse','surgeon','paramedic','healthcare','icu']):
        recs.append({
            'icon'    : '👨‍⚕️',
            'title'   : 'Healthcare Worker Burnout Prevention',
            'desc'    : (
                'Structured debriefing after difficult cases, '
                'peer support groups, and mandatory rest periods '
                'are clinically proven to reduce burnout.'),
            'priority': 'high'
        })
    elif any(k in prof_lower for k in
             ['developer','engineer','programmer','software','tech']):
        recs.append({
            'icon'    : '💻',
            'title'   : 'Tech Worker Digital Detox',
            'desc'    : (
                'Screen fatigue elevates cortisol. '
                'Take a 20-min screen-free break every 2 hours. '
                'Avoid screens 1 hour before sleep.'),
            'priority': 'medium'
        })
    elif any(k in prof_lower for k in
             ['teacher','professor','lecturer','educator']):
        recs.append({
            'icon'    : '📚',
            'title'   : 'Educator Boundary Setting',
            'desc'    : (
                'Emotional labour in teaching is often invisible. '
                'Set firm work-home boundaries and '
                'practice emotional offloading through journaling.'),
            'priority': 'medium'
        })
    elif any(k in prof_lower for k in
             ['lawyer','attorney','judge','legal']):
        recs.append({
            'icon'    : '⚖️',
            'title'   : 'Legal Professional Stress Protocol',
            'desc'    : (
                'Legal professionals have 3.6× higher depression rates. '
                'Structured mindfulness and case-load management '
                'are evidence-based interventions.'),
            'priority': 'high'
        })
    elif any(k in prof_lower for k in
             ['student','intern','trainee']):
        recs.append({
            'icon'    : '🎓',
            'title'   : 'Student Mental Wellness',
            'desc'    : (
                'Academic pressure peaks before exams. '
                'Use spaced repetition to reduce study stress, '
                'and reach out to campus counseling services.'),
            'priority': 'medium'
        })
    else:
        recs.append({
            'icon'    : '💼',
            'title'   : 'Workplace Stress Management',
            'desc'    : (
                f'As a {profession}, identify your top 3 stressors '
                f'and address one per week. '
                f'Small changes compound into big improvements.'),
            'priority': 'medium'
        })

    # ── 3. Age-specific ───────────────────────────────────────
    if age_group == 'teen':
        recs.append({
            'icon'    : '🧒',
            'title'   : 'Teen Brain Stress Support',
            'desc'    : (
                'At your age, social connection is the #1 protective factor. '
                'Prioritise in-person time with trusted friends '
                'and family over social media.'),
            'priority': 'medium'
        })
    elif age_group == 'young_adult':
        recs.append({
            'icon'    : '🌱',
            'title'   : 'Build Resilience Habits Early',
            'desc'    : (
                'Your 20s are the best time to establish mental health habits. '
                'Regular exercise, consistent sleep schedule, and a '
                'mindfulness practice now pays dividends for decades.'),
            'priority': 'low'
        })
    elif age_group == 'middle_age':
        recs.append({
            'icon'    : '⚡',
            'title'   : 'Mid-Life Stress Checkpoint',
            'desc'    : (
                'Burnout peaks in the 40s. Schedule quarterly mental health '
                'check-ins and consider reassessing career priorities '
                'and work-life balance.'),
            'priority': 'medium'
        })
    elif age_group in ('senior', 'elderly'):
        recs.append({
            'icon'    : '🌿',
            'title'   : 'Senior Wellbeing Strategies',
            'desc'    : (
                'Social engagement, light daily exercise, and cognitive '
                'stimulation (reading, puzzles) are the most effective '
                f'tools for mental health at {age}.'),
            'priority': 'medium'
        })

    # ── 4. Gender-specific ────────────────────────────────────
    if gender.lower() == 'female':
        recs.append({
            'icon'    : '🌸',
            'title'   : 'Hormonal & Stress Cycle Awareness',
            'desc'    : (
                'Track stress levels alongside your cycle — '
                'luteal phase increases cortisol sensitivity. '
                'Magnesium supplementation and yoga show strong '
                'evidence for female stress reduction.'),
            'priority': 'medium'
        })
    elif gender.lower() == 'male':
        recs.append({
            'icon'    : '💪',
            'title'   : 'Break the Silence on Stress',
            'desc'    : (
                'Men are 3× less likely to seek help for mental health. '
                'Physical activity, particularly team sports, provides '
                'both stress relief and social connection.'),
            'priority': 'medium'
        })

    # ── 5. Working hours ──────────────────────────────────────
    if hours_cat in ('excessive', 'extreme'):
        recs.append({
            'icon'    : '⏰',
            'title'   : f'Critical: Reduce Working Hours ({hours}h/day)',
            'desc'    : (
                f'Working {hours}h/day puts you at significantly elevated '
                f'risk. Even reducing by 1–2 hours/day produces measurable '
                f'HRV improvement within 2 weeks.'),
            'priority': 'high'
        })
    elif hours_cat == 'overworked':
        recs.append({
            'icon'    : '🕐',
            'title'   : 'Implement Work Boundaries',
            'desc'    : (
                f'At {hours}h/day, use time-blocking to protect personal time. '
                f'Set a firm "shutdown ritual" at end of workday.'),
            'priority': 'medium'
        })

    # ── 6. BMI-based ─────────────────────────────────────────
    height  = profile.get('height_cm', 0) or 0
    weight  = profile.get('weight_kg', 0) or 0
    bmi_val, bmi_cat, bmi_tier = calculate_bmi(height, weight)

    if bmi_val:
        if bmi_val >= 30:
            recs.append({
                'icon'    : '⚖️',
                'title'   : f'BMI {bmi_val} — Metabolic Health Priority',
                'desc'    : (
                    f'Obesity (BMI {bmi_val}) is linked to a 55% higher '
                    f'risk of depression and significantly reduces HRV. '
                    f'Even a 5–10% weight reduction produces measurable '
                    f'mental health improvements within 3 months.'),
                'priority': 'high'
            })
        elif bmi_val >= 25:
            recs.append({
                'icon'    : '🥗',
                'title'   : f'BMI {bmi_val} — Healthy Weight Goal',
                'desc'    : (
                    f'Overweight BMI moderately elevates cortisol. '
                    f'Mediterranean diet + 150 min/week exercise is '
                    f'the most evidence-based path to both weight '
                    f'and mental health improvement.'),
                'priority': 'medium'
            })
        elif bmi_val < 18.5:
            recs.append({
                'icon'    : '🍽️',
                'title'   : f'BMI {bmi_val} — Nutritional Support Needed',
                'desc'    : (
                    'Underweight BMI is associated with nutrient '
                    'deficiencies that directly impair serotonin and '
                    'dopamine production. Consider a dietitian '
                    'consultation alongside mental health support.'),
                'priority': 'high'
            })
        else:
            recs.append({
                'icon'    : '✨',
                'title'   : f'BMI {bmi_val} — Maintain Healthy Weight',
                'desc'    : (
                    'Your healthy BMI is a strong protective factor. '
                    'Regular exercise that maintains this range also '
                    "boosts BDNF — the brain's natural antidepressant."),
                'priority': 'low'
            })

    # ── 7. Health issues ──────────────────────────────────────
    health_lower = health.lower()
    if any(c in health_lower for c in ['anxiety','panic','ocd','ptsd']):
        recs.append({
            'icon'    : '🧘',
            'title'   : 'Anxiety Management Protocol',
            'desc'    : (
                'CBT (Cognitive Behavioural Therapy) and diaphragmatic '
                'breathing (4-7-8 technique) are first-line evidence-based '
                'treatments for anxiety. Consider a licensed therapist.'),
            'priority': 'high'
        })
    if any(c in health_lower for c in ['depression','bipolar','mood']):
        recs.append({
            'icon'    : '💊',
            'title'   : 'Mood Disorder Support',
            'desc'    : (
                'Consistent medication adherence, sleep regulation, and '
                'light therapy in the morning are clinically proven '
                'to stabilise mood disorders.'),
            'priority': 'high'
        })
    if any(c in health_lower for c in
           ['hypertension','heart','cardiac','blood pressure']):
        recs.append({
            'icon'    : '❤️',
            'title'   : 'Cardiovascular-Mental Health Link',
            'desc'    : (
                'Your cardiac condition and mental health are '
                'bidirectionally linked. Stress management directly '
                'improves cardiovascular outcomes — coordinate with '
                'your cardiologist.'),
            'priority': 'high'
        })
    if any(c in health_lower for c in
           ['insomnia','sleep','tired','fatigue']):
        recs.append({
            'icon'    : '😴',
            'title'   : 'Sleep Restoration Plan',
            'desc'    : (
                'CBT-I (for insomnia) is more effective than medication '
                'long-term. Maintain strict sleep-wake times, '
                'even on weekends.'),
            'priority': 'high'
        })
    if any(c in health_lower for c in ['diabetes','thyroid','hormone']):
        recs.append({
            'icon'    : '🩺',
            'title'   : 'Metabolic-Mental Health Monitoring',
            'desc'    : (
                'Metabolic conditions affect neurotransmitter balance. '
                'Ensure your blood markers are regularly checked '
                'alongside mental health monitoring.'),
            'priority': 'medium'
        })

    # ── 8. Universal ─────────────────────────────────────────
    recs.append({
        'icon'    : '🏃',
        'title'   : 'Daily Physical Activity',
        'desc'    : (
            '30 min moderate exercise 5×/week improves HRV by '
            '15–20% and reduces EEG stress markers. '
            'Even a brisk walk counts.'),
        'priority': 'low'
    })
    recs.append({
        'icon'    : '😴',
        'title'   : 'Sleep Quality Optimisation',
        'desc'    : (
            '7–9 hours of consistent sleep is the single most powerful '
            'mental health intervention. Poor sleep worsens both '
            'EEG coherence and HRV.'),
        'priority': 'low'
    })

    # Sort by priority and return top 7
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recs.sort(key=lambda x: priority_order.get(x['priority'], 3))
    return recs[:7]


# ─────────────────────────────────────────────────────────────
# PEER COMPARISON
# ─────────────────────────────────────────────────────────────
def get_peer_context(risk_level, profile):
    age_group = get_age_group(profile.get('age', 30))
    prof_tier = get_profession_tier(profile.get('profession', ''))
    hours_cat = get_hours_category(profile.get('avg_working_hours', 8))

    group_risk_rates = {
        ('teen',        'high'): 32,
        ('young_adult', 'high'): 28,
        ('adult',       'high'): 24,
        ('middle_age',  'high'): 29,
        ('senior',      'high'): 22,
        ('elderly',     'high'): 20,
    }
    prof_baseline  = {'high': 38, 'medium': 24, 'low': 16}
    hours_baseline = {
        'extreme': 45, 'excessive': 36, 'overworked': 27,
        'healthy': 18, 'underworked': 20,
    }

    age_rate  = group_risk_rates.get((age_group, 'high'), 25)
    prof_rate = prof_baseline.get(prof_tier, 24)
    hour_rate = hours_baseline.get(hours_cat, 22)

    if risk_level == 0:
        comparison = (
            f"You are in BETTER shape than approximately "
            f"{prof_rate}% of {profile.get('profession','people')} "
            f"who show high-risk indicators."
        )
    elif risk_level == 1:
        comparison = (
            f"Medium risk is common in your group — roughly "
            f"{age_rate}% of {AGE_GROUP_LABELS[age_group]} "
            f"experience similar stress levels."
        )
    else:
        comparison = (
            f"High risk affects approximately {hour_rate}% of people "
            f"working {profile.get('avg_working_hours','similar')} "
            f"hours/day. You are not alone — but action is needed."
        )

    return {
        'comparison'      : comparison,
        'age_group_label' : AGE_GROUP_LABELS[age_group],
        'profession_tier' : prof_tier,
        'hours_category'  : HOURS_LABELS[hours_cat],
        'age_risk_rate'   : age_rate,
        'prof_risk_rate'  : prof_rate,
        'hours_risk_rate' : hour_rate,
    }


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def personalize(result, profile, high_count=0, total=0):
    """
    Takes raw predict_risk() result + user profile dict.
    Returns enriched result with personalised context.
    """
    # ── 1. Soft label nudge first ─────────────────────────────
    nudge = soft_label_nudge(
        result['prob_low']    / 100,
        result['prob_medium'] / 100,
        result['prob_high']   / 100,
        profile
    )
    result['prob_low']    = nudge['prob_low']
    result['prob_medium'] = nudge['prob_medium']
    result['prob_high']   = nudge['prob_high']
    result['risk_level']  = nudge['risk_level']
    result['risk_name']   = nudge['risk_name']
    result['nudged']      = nudge['nudged']

    risk_level = result['risk_level']
    risk_name  = result['risk_name']
    confidence = result['confidence']

    # ── 2. Adjusted confidence ────────────────────────────────
    adj_conf, personal_factors = adjust_confidence(
        confidence, risk_level, profile)

    # ── 3. Interpretation ─────────────────────────────────────
    interpretation = get_risk_interpretation(
        risk_level, risk_name, profile,
        adj_conf, personal_factors)

    # ── 4. Recommendations ────────────────────────────────────
    recommendations = get_recommendations(
        risk_level, profile, high_count, total)

    # ── 5. Peer context ───────────────────────────────────────
    peer_context = get_peer_context(risk_level, profile)

    # ── 6. BMI summary ────────────────────────────────────────
    bmi_val, bmi_cat, bmi_tier = calculate_bmi(
        profile.get('height_cm', 0) or 0,
        profile.get('weight_kg', 0) or 0
    )

    # ── 7. Enrich result ──────────────────────────────────────
    result['adjusted_confidence'] = adj_conf
    result['confidence_delta']    = round(adj_conf - confidence, 1)
    result['personal_factors']    = personal_factors
    result['interpretation']      = interpretation
    result['recommendations']     = recommendations
    result['peer_context']        = peer_context
    result['bmi']                 = bmi_val
    result['bmi_category']        = bmi_cat
    result['bmi_tier']            = bmi_tier
    result['profile_summary']     = {
        'age_group' : AGE_GROUP_LABELS[get_age_group(
                          profile.get('age', 30))],
        'prof_tier' : get_profession_tier(
                          profile.get('profession', '')).title(),
        'hours_cat' : HOURS_LABELS[get_hours_category(
                          profile.get('avg_working_hours', 8))],
        'bmi'       : bmi_val,
        'bmi_cat'   : bmi_cat,
    }

    return result