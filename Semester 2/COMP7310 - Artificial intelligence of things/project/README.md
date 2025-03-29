# MQTT Flask Dashboard

This project is a simple MQTT-based dashboard built using Flask, Flask-SocketIO, and SQLite. It allows real-time monitoring of MQTT messages and stores them in a database for historical reference.

## Project Structure

```
.gitignore
README.md
backend/
    app.py
    mqtt_messages.db
    templates/
        index.html
esp32/
```

### Key Components

- **`backend/app.py`**: The main Flask application that handles MQTT communication, database operations, and serves the web interface.
- **`backend/templates/index.html`**: The front-end dashboard for displaying MQTT messages in real-time.
- **`backend/mqtt_messages.db`**: SQLite database for storing MQTT messages.
- **`esp32/`**: Placeholder for ESP32-related code (not included in this repository).

## Features

1. **Real-time MQTT Message Display**:
   - Subscribes to the topic `flask/esp32/bjh` on the MQTT broker `broker.emqx.io`.
   - Displays incoming messages in real-time on the web dashboard.

2. **Message Persistence**:
   - Stores MQTT messages in an SQLite database (`mqtt_messages.db`).
   - Provides an endpoint to fetch the last 20 messages.

3. **Web Interface**:
   - A simple dashboard built with HTML and JavaScript to display messages and interact with the backend.

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- MQTT broker (e.g., [EMQX](https://www.emqx.io/))

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python -c "from app import init_db; init_db()"
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Access the dashboard:
   Open your browser and navigate to `http://localhost:5000`.

## Usage

- **Real-time Messages**: View live MQTT messages on the dashboard.
- **Historical Messages**: Scroll through the last 20 messages stored in the database.

## File Descriptions

- **`app.py`**:
  - Initializes the SQLite database.
  - Handles MQTT communication using the `paho-mqtt` library.
  - Provides Flask routes for the web interface and API endpoints.

- **`index.html`**:
  - Displays MQTT messages in real-time using Socket.IO.
  - Fetches historical messages via the `/messages` endpoint.

## Example MQTT Message Flow

1. ESP32 publishes a message to the topic `flask/esp32/bjh`.
2. The Flask app receives the message via the MQTT broker.
3. The message is saved to the SQLite database.
4. The message is broadcast to the web dashboard using Socket.IO.

## Future Enhancements

- Add authentication for the MQTT broker and web interface.
- Expand the ESP32 codebase for IoT device integration.
- Implement advanced message filtering and analytics.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [Socket.IO](https://socket.io/)
- [Paho MQTT](https://www.eclipse.org/paho/)
- [SQLite](https://www.sqlite.org/)
