const canvas = document.getElementById('starfield');
const ctx = canvas.getContext('2d');

let width = canvas.width = window.innerWidth;
let height = canvas.height = window.innerHeight;

const stars = [];
const numStars = 200;

function Star() {
    this.x = Math.random() * width;
    this.y = Math.random() * height;
    this.z = Math.random() * width;
    this.size = 0.5;
    this.speed = 0.05;
}

Star.prototype.move = function() {
    this.z -= this.speed;
    if (this.z <= 0) {
        this.z = width;
    }
};

Star.prototype.draw = function() {
    const x = (this.x - width / 2) * (width / this.z);
    const y = (this.y - height / 2) * (width / this.z);
    const s = this.size * (width / this.z);

    ctx.beginPath();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.arc(x + width / 2, y + height / 2, s, 0, 2 * Math.PI);
    ctx.fill();
};

for (let i = 0; i < numStars; i++) {
    stars.push(new Star());
}

function animate() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, width, height);

    stars.forEach(star => {
        star.move();
        star.draw();
    });

    requestAnimationFrame(animate);
}

animate();

window.addEventListener('resize', function() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
});

canvas.addEventListener('mousemove', function(e) {
    const mouseX = e.clientX;
    const mouseY = e.clientY;

    stars.forEach(star => {
        const dx = (mouseX - width / 2) / 100;
        const dy = (mouseY - height / 2) / 100;
        star.x += dx;
        star.y += dy;
    });
});