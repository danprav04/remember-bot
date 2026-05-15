"""
Static HTML pages for the Remember Bot web presence.
"""

PRIVACY_POLICY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy | Remember Bot</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-primary: #38bdf8;
            --accent-secondary: #818cf8;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(129, 140, 248, 0.15) 0%, transparent 50%);
        }

        .container {
            max-width: 800px;
            width: 100%;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }

        h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            color: var(--accent-primary);
        }

        p {
            margin-bottom: 1.25rem;
            color: var(--text-muted);
        }

        .highlight {
            color: var(--text-main);
            font-weight: 500;
        }

        ul {
            list-style: none;
            margin-bottom: 1.5rem;
        }

        li {
            position: relative;
            padding-left: 1.5rem;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
        }

        li::before {
            content: "•";
            position: absolute;
            left: 0;
            color: var(--accent-secondary);
            font-weight: bold;
        }

        footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--glass-border);
            text-align: center;
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        .badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent-primary);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }

        @media (max-width: 640px) {
            .container {
                padding: 2rem;
            }
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="badge">Last Updated: May 2026</div>
        <h1>Privacy Policy</h1>
        
        <p>Your privacy is paramount. <span class="highlight">Remember Bot</span> is designed to be a personal, secure companion for your memories. This policy explains how we handle your information.</p>

        <h2>1. Information We Collect</h2>
        <p>We only collect information that you explicitly send to the bot. This includes:</p>
        <ul>
            <li>Text messages and facts you ask us to remember.</li>
            <li>Voice messages (transcribed for processing and storage).</li>
            <li>Photos (analyzed for visual content to store as text descriptions).</li>
            <li>Basic metadata provided by the messaging platform (Telegram) such as your User ID and name.</li>
        </ul>

        <h2>2. How We Use Your Data</h2>
        <p>Your data is used <span class="highlight">exclusively</span> to provide the memory service. We use it to:</p>
        <ul>
            <li>Retrieve facts when you ask questions.</li>
            <li>Provide context to the AI model for more relevant conversations.</li>
            <li>Generate personal statistics and data exports upon your request.</li>
        </ul>
        <p>We do <span class="highlight">not</span> use your data for advertising, nor do we sell it to third parties. Your data is not used to train global AI models.</p>

        <h2>3. Data Storage & Security</h2>
        <p>All data is stored in a secured database with industry-standard encryption. We use specialized "vector embeddings" to store your memories, which are mathematical representations that are meaningless without the proper context and authorization.</p>

        <h2>4. Your Rights and Control</h2>
        <p>You have full control over your data directly within the bot interface:</p>
        <ul>
            <li><strong>View:</strong> Use <code>/facts</code> to see everything we've remembered.</li>
            <li><strong>Delete:</strong> Use <code>/forget</code> to remove specific items or wipe all your data.</li>
            <li><strong>Export:</strong> Use <code>/export</code> to download a full copy of your data in JSON and Markdown formats.</li>
        </ul>

        <h2>5. Third-Party Services</h2>
        <p>We process your messages using AI models (such as Google Gemini). These providers receive the text of your messages to generate responses, but they are bound by enterprise privacy agreements that prevent them from using your personal data for their own purposes.</p>

        <footer>
            <p>&copy; 2026 Remember Bot. Built with privacy by design.</p>
        </footer>
    </div>
</body>
</html>
"""
