function toggleRunning() {
    const button = document.querySelector('.stButton button');
    if (button.innerText === '▶️ Start') {
        button.innerText = '⏹️ Stop';
    } else {
        button.innerText = '▶️ Start';
    }
}