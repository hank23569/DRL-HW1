from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
n = 5

@app.route("/", methods=["GET", "POST"])
def index():
    global n
    n = 5
    if request.method == "POST":
        try:
            n = int(request.form["size"])
            if not (5 <= n <= 9):
                n = 5

        except:
            n = 5
    return render_template("index.html", n=n)

@app.route("/show_policy", methods=["POST"])
def show_policy():
    grid_data = request.get_json()
    start = tuple(grid_data["start"])
    end = tuple(grid_data["end"])
    blocks = [tuple(b) for b in grid_data["blocks"]]

    policy, value = compute_policy_and_value(n, start, end, blocks)
    policy_img = plot_matrix(policy, blocks, "policy")
    value_img = plot_matrix(value, blocks, "value")

    return jsonify({
        "policy_img": policy_img,
        "value_img": value_img
    })

def compute_policy_and_value(n, start, goal, blocks):
    actions = ['↑', '↓', '←', '→']
    policy = np.random.choice(actions, size=(n, n))

    action_map = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1)
    }

    reward = np.full((n, n), -1.0)
    reward[goal] = 1.0

    value = np.zeros((n, n))
    gamma = 0.9
    theta = 1e-4

    while True:
        delta = 0
        new_v = value.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) in blocks:
                    continue
                a = policy[i][j]
                di, dj = action_map[a]
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in blocks:
                    v = reward[i][j] + gamma * value[ni][nj]
                else:
                    v = reward[i][j]
                new_v[i][j] = v
                delta = max(delta, abs(value[i][j] - v))
        value = new_v
        if delta < theta:
            break
    return policy, value

def plot_matrix(data, blocks, mode):
    fig, ax = plt.subplots()
    n = data.shape[0]
    for i in range(n):
        for j in range(n):
            if (i, j) in blocks:
                ax.add_patch(plt.Rectangle((j, n - i - 1), 1, 1, color='gray'))
            else:
                txt = data[i][j] if mode == "policy" else f"{data[i][j]:.2f}"
                ax.text(j + 0.5, n - i - 1 + 0.5, txt, ha='center', va='center')
    ax.set_xticks(np.arange(n+1))
    ax.set_yticks(np.arange(n+1))
    ax.grid(True)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_title(f"{mode.capitalize()} Matrix")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str

if __name__ == "__main__":
    app.run(debug=True)
