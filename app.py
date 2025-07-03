import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import face_recognition
import datetime
import pickle
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DATABASE_FILE = 'face_data.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            try:
                db_data = pickle.load(f)
                for user in db_data['users']:
                    if isinstance(user['descriptors'], str):
                        user['descriptors'] = json.loads(user['descriptors'])
                return db_data
            except (EOFError, pickle.UnpicklingError) as e:
                return {
                    'users': [],
                    'attendance': {}
                }
            except Exception as e:
                return {
                    'users': [],
                    'attendance': {}
                }
    return {
        'users': [],
        'attendance': {}
    }

def save_database(db):
    db_to_save = db.copy()
    db_to_save['users'] = []
    for user in db['users']:
        user_copy = user.copy()
        user_copy['descriptors'] = json.dumps(user_copy['descriptors']) 
        db_to_save['users'].append(user_copy)

    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(db_to_save, f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get('id')
    user_name = request.form.get('name')
    descriptors_json = request.form.get('descriptors')

    if not user_id or not user_name or not descriptors_json:
        return jsonify({'error': 'Missing ID, name, or descriptors'}), 400

    try:
        descriptors_list = json.loads(descriptors_json) 
        if not isinstance(descriptors_list, list) or not all(isinstance(d, list) and len(d) == 128 for d in descriptors_list):
            raise ValueError("Descriptors must be a list of 128-element lists.")
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({'error': f'Invalid descriptors format: {e}'}), 400

    db = load_database()

    for user in db['users']:
        if user['id'] == user_id:
            return jsonify({'error': f'User with Roll No "{user_id}" already registered. Cannot re-register.'}), 409

    user_data = {
        'id': user_id,
        'name': user_name,
        'descriptors': descriptors_list, 
        'registration_date': datetime.datetime.now().isoformat()
    }
    db['users'].append(user_data)
    
    save_database(db) 

    return jsonify({'message': 'User registered successfully', 'user_id': user_id, 'user_name': user_name})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image file'}), 400

    db = load_database() 

    known_face_descriptors_np = []
    known_face_labels = []
    
    for user_data in db['users']:
        descriptors_list_for_user = user_data['descriptors'] 
        for descriptor_array in descriptors_list_for_user:
            known_face_descriptors_np.append(np.array(descriptor_array))
            known_face_labels.append({'id': user_data['id'], 'name': user_data['name']})


    if len(known_face_descriptors_np) == 0:
        return jsonify({'recognized': False, 'message': 'No registered faces available for recognition'}), 200

    filename = secure_filename(f"temp_recognize_{datetime.datetime.now().timestamp()}.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return jsonify({'recognized': False, 'message': 'No face detected in the image'}), 200
        
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        matches = face_recognition.compare_faces(known_face_descriptors_np, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_descriptors_np, face_encoding)
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            recognized_info = known_face_labels[best_match_index]
            user_id = recognized_info['id']
            user_name = recognized_info['name']

            today = datetime.date.today().isoformat()
            
            if user_id not in db['attendance']:
                db['attendance'][user_id] = []
            
            message = 'Attendance already recorded today'
            if today not in db['attendance'][user_id]:
                db['attendance'][user_id].append(today)
                save_database(db)
                message = 'Attendance recorded'
            
            return jsonify({
                'recognized': True,
                'message': message,
                'user_id': user_id,
                'user_name': user_name,
                'distance': float(face_distances[best_match_index]),
                'attendance_dates': db['attendance'].get(user_id, [])
            })
        else:
            return jsonify({'recognized': False, 'message': 'User not recognized'}), 200

    except Exception as e:
        import traceback
        return jsonify({'error': f'Recognition failed due to server error: {e}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/users_data', methods=['GET'])
def get_users_data():
    try:
        db = load_database()
        users_list = []
        
        for user in db['users']:
            user_data = user.copy()
            user_id = user_data.get('id')
            user_data['attendance_count'] = len(db['attendance'].get(user_id, [])) if user_id else 0
            user_data['attendance_dates'] = db['attendance'].get(user_id, []) 
            users_list.append(user_data)
        
        today = datetime.date.today().isoformat()
        present_today_ids = set()
        for user_id, dates in db['attendance'].items():
            if today in dates:
                present_today_ids.add(user_id)
        
        all_registered_ids = set(user['id'] for user in db['users'])
        absent_today_ids = all_registered_ids - present_today_ids

        present_names = []
        absent_names = []
        
        id_to_name_map = {user['id']: user['name'] for user in db['users']}

        for user_id in present_today_ids:
            present_names.append(id_to_name_map.get(user_id, f"Unknown User ({user_id})"))
        
        for user_id in absent_today_ids:
            absent_names.append(id_to_name_map.get(user_id, f"Unknown User ({user_id})"))

        return jsonify({
            'users': users_list,
            'daily_summary': {
                'total_registered': len(all_registered_ids),
                'present_count': len(present_today_ids),
                'absent_count': len(absent_today_ids),
                'present_names': sorted(present_names),
                'absent_names': sorted(absent_names),
                'present_today_ids': list(present_today_ids) 
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'error': f'Server error fetching user data: {e}'}), 500


@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    try:
        db = load_database()
        db['attendance'] = {} 
        save_database(db)
        return jsonify({'message': 'All attendance records have been reset successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to reset attendance: {e}'}), 500

@app.route('/reset_daily_attendance', methods=['POST'])
def reset_daily_attendance():
    date_to_reset = request.json.get('date')
    if not date_to_reset:
        return jsonify({'error': 'Missing date parameter'}), 400
    
    try:
        db = load_database()
        attendance_cleared_count = 0
        for user_id in list(db['attendance'].keys()):
            if date_to_reset in db['attendance'][user_id]:
                db['attendance'][user_id].remove(date_to_reset)
                attendance_cleared_count += 1
        save_database(db)
        return jsonify({'message': f'Attendance for {date_to_reset} has been reset for {attendance_cleared_count} users.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to reset daily attendance: {e}'}), 500

@app.route('/reset_all_data', methods=['POST'])
def reset_all_data():
    try:
        db = {
            'users': [],
            'attendance': {}
        }
        save_database(db)
        return jsonify({'message': 'All registered users and attendance records have been reset successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to reset all data: {e}'}), 500


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
