from flask import Flask, render_template, request, jsonify
import sqlite3, subprocess
from datetime import datetime
import get_faces_from_camera_tkinter
import features_extraction_to_csv
import attendance_taker


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/register')
def register():
    # Import and execute your Python script here
    try:
        result = subprocess.run(['python', 'get_faces_from_camera_tkinter.py'], capture_output=True, text=True)
  # Call a function or execute code in your script
        return f'Script executed successfully. Result: {result}'
    except Exception as e:
        return f'Error executing script: {str(e)}'

@app.route('/train')
def train():
    # Import and execute your Python script here
    try:
        result = subprocess.run(['python', 'features_extraction_to_csv.py'], capture_output=True, text=True)
  # Call a function or execute code in your script
        return f'Script executed successfully. Result: {result}'
    except Exception as e:
        return f'Error executing script: {str(e)}'


@app.route('/take_attendance')
def take_attendance():
    try:
        # Run your script for taking attendance
        result = subprocess.run(['python', 'liveness_detection.py'], capture_output=True, text=True)
        

        print(result)
        # Get the output of your script
        output = result.stdout

        # print(output)

        # Return JSON response
        return jsonify(result=output)
    except Exception as e:
        # Return JSON response in case of error
        return jsonify(error=str(e))


@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()
    columns = [description[0] for description in cursor.description]


    conn.close()
     # Specify the CSV file path
    import csv
    csv_file_path = f'{selected_date}_attendee_list.csv'

    # Write data to CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow(columns)
        
        # Write the data
        csv_writer.writerows(attendance_data)

    print(f'Data exported to {csv_file_path}')


    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

if __name__ == '__main__':
    app.run(app.run(port=8000))
