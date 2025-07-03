# 📘 Live Face Recognition Attendance System

A Python Flask web app for face registration, recognition, and attendance tracking using 128D face descriptors. All data is stored locally using Pickle.

---

## 🧾 Table of Contents

- Features  
- Installation  
- Project Structure  
- How It Works  
- API Endpoints  
- Usage  
- Examples  
- Todo  
- License  

---

## 🚀 Features

- Register users with face descriptors (128D)  
- Recognize users from uploaded images  
- Mark daily attendance automatically  
- View present and absent lists  
- Reset daily or complete attendance records  
- REST API with JSON responses  
- Simple web UI (`index.html`)  

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rahitya86/Live-Face-Recognition.git
cd Live-Face-Recognition
```

### 2. Set up a virtual environment

```bash
python -m venv venv
```

Then activate it:

- On Windows:  
  `venv\Scripts\activate`

- On macOS/Linux:  
  `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, run:

```bash
pip install flask flask-cors face_recognition opencv-python numpy
```

---

## 🏗️ Project Structure

```
Live-Face-Recognition/
├── app.py                    # Main backend logic
├── templates/index.html      # Web interface
├── static/models/            # Face recognition models
├── uploads/                  # Temporary image uploads
├── face_data.pkl             # Stores user + attendance info
├── .vscode/                  # VSCode launch configs
└── README.md
```

---

## 🧠 How It Works

- You register a user by submitting their ID, name, and face descriptors.
- Descriptors are saved in `face_data.pkl`.
- When a face image is uploaded:
  - It’s encoded to a 128D vector.
  - Compared with all saved users.
  - If matched, the attendance is marked.
- Attendance is saved by date and can be reset.

---

## 📡 API Endpoints

### 🔹 POST `/register_face`

Registers a new user with their name, ID, and face descriptors.

**Form fields:**

- `id`: User ID  
- `name`: User name  
- `descriptors`: JSON string of 128D arrays  

Example:

```json
[[0.1, 0.2, ..., 0.99]]
```

---

### 🔹 POST `/recognize_face`

Recognizes and marks attendance for a face in an uploaded image.

**Form field:**

- `image`: Image file (jpg, jpeg, png)

**Success response:**

```json
{
  "recognized": true,
  "message": "Attendance recorded",
  "user_id": "123",
  "user_name": "Alice",
  "attendance_dates": ["2025-07-03"]
}
```

---

### 🔹 GET `/users_data`

Returns user list and attendance summary.

**Includes:**

- Total registered  
- Present today  
- Absent today  
- Attendance history for each user  

---

### 🔹 POST `/reset_attendance`

Resets **all** attendance records (but keeps user data).

---

### 🔹 POST `/reset_daily_attendance`

Resets attendance **only for a specific date**.

**JSON body:**

```json
{ "date": "2025-07-03" }
```

---

### 🔹 POST `/reset_all_data`

Deletes **all** users and attendance records.

---

## 🧪 Usage

### Start the Flask app:

```bash
python app.py
```

### Visit:

```
http://localhost:5000
```

Use Postman or browser to test endpoints.

---

## 📸 Examples

### Register user

Use POST `/register_face` with:

- ID: `"001"`
- Name: `"John Doe"`
- Descriptors: `[[...128 values...]]`

---

### Recognize user

Send an image to POST `/recognize_face`.

Returns:

- Match result  
- Name and ID  
- Attendance status  

---

## ✅ Todo

- [ ] Export attendance to CSV  
- [ ] Add face capture via webcam  
- [ ] Use SQLite instead of `.pkl`  
- [ ] Admin dashboard for user control  

---

## 📄 License

MIT License. Free for personal and academic use.

---
