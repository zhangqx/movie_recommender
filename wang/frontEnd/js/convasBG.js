var canvasGranuleBg = (function(){

    var canvasCtx2dOptions = {

        initCanvas:function(id,win){
            var canvas;
            if(id){
                canvas = document.getElementById(id);
            }else{
                var body = document.getElementsByTagName('body')[0];
                body.innerHTML = body.innerHTML + '<canvas id="canvasGranuleBg" width="2880" height="1800" style="position: absolute;top: 0;left: 0; z-index: -1;"></canvas>';
                canvas = document.getElementById('canvasGranuleBg');
            }
            var ctx = canvas.getContext("2d");
            ctx.canvas = canvas;

            return ctx;
        },
        resize : function() {
            window.ctx2d.canvas.width = document.body.scrollWidth || window.innerWidth || document.body.clientWidth || document.documentElement.clientWidth;
            window.ctx2d.canvas.height =  document.body.scrollHeight || window.innerHeight || document.body.clientHeight || document.documentElement.clientHeight;
        },
        mousemove : function(e) {
            window.ctx2d.warea.x = e.clientX + document.documentElement.scrollLeft;
            window.ctx2d.warea.y = e.clientY + document.documentElement.scrollTop;
        },
        mouseout : function(e) {
            window.ctx2d.warea.x = null;
            window.ctx2d.warea.y = null;
        },
        animate :  function() {
            window.ctx2d.clearRect(0, 0, window.ctx2d.canvas.width, window.ctx2d.canvas.height);

            // 将鼠标坐标添加进去，产生一个用于比对距离的点数组
            window.ctx2d.ndots = [window.ctx2d.warea].concat(window.ctx2d.dots);

            window.ctx2d.dots.forEach(function(dot) {

                // 粒子位移
                dot.x += dot.xa;
                dot.y += dot.ya;

                // 遇到边界将加速度反向
                dot.xa *= (dot.x > window.ctx2d.canvas.width || dot.x < 0) ? -1 : 1;
                dot.ya *= (dot.y > window.ctx2d.canvas.height || dot.y < 0) ? -1 : 1;

                // 绘制点
                window.ctx2d.fillRect(dot.x - 0.5, dot.y - 0.5, 1, 1);

                // 循环比对粒子间的距离
                for (var i = 0; i < window.ctx2d.ndots.length; i++) {
                    var d2 = window.ctx2d.ndots[i];

                    if (dot === d2 || d2.x === null || d2.y === null) continue;

                    var xc = dot.x - d2.x;
                    var yc = dot.y - d2.y;

                    // 两个粒子之间的距离
                    var dis = xc * xc + yc * yc;

                    // 距离比
                    var ratio;

                    // 如果两个粒子之间的距离小于粒子对象的max值，则在两个粒子间画线
                    if (dis < d2.max) {

                        // 如果是鼠标，则让粒子向鼠标的位置移动
                        if (d2 === window.ctx2d.warea && dis > (d2.max / 2)) {
                            dot.x -= xc * 0.03;
                            dot.y -= yc * 0.03;
                        }

                        // 计算距离比
                        ratio = (d2.max - dis) / d2.max;

                        // 画线
                        window.ctx2d.beginPath();
                        window.ctx2d.lineWidth = ratio / 2;
                        window.ctx2d.strokeStyle = 'rgba(0,0,0,' + (ratio + 0.2) + ')';
                        window.ctx2d.moveTo(dot.x, dot.y);
                        window.ctx2d.lineTo(d2.x, d2.y);
                        window.ctx2d.stroke();
                    }
                }
                // 将已经计算过的粒子从数组中删除
                window.ctx2d.ndots.splice(window.ctx2d.ndots.indexOf(dot), 1);
            });
            window.ctx2d.RAF(window.ctx2d.animate);
        }
    }


    function newInstanceCanvs(id){
        window.ctx2d = canvasCtx2dOptions.initCanvas(id,window);
        window.ctx2d.resize = canvasCtx2dOptions.resize;
        window.ctx2d.warea = {x: null, y: null, max: 20000};
        window.ctx2d.animate = canvasCtx2dOptions.animate;

        window.ctx2d.resize();

        window.ctx2d.RAF = function(callback) {
            window.setTimeout(callback, 1000 / 60);
        };
        window.addEventListener('mousemove',canvasCtx2dOptions.mousemove);
        window.addEventListener('mouseout',canvasCtx2dOptions.mouseout);
        window.addEventListener('resize',window.ctx2d.resize);

        //添加粒子 x，y为粒子坐标，xa, ya为粒子xy轴加速度，max为连线的最大距离
        var dots = [];
        for (var i = 0; i < 300; i++) {
            var x = Math.random() * ctx2d.canvas.width;
            var y = Math.random() * ctx2d.canvas.height;
            var xa = Math.random() * 2 - 1;
            var ya = Math.random() * 2 - 1;

            dots.push({
                x: x,
                y: y,
                xa: xa,
                ya: ya,
                max: 6000
            })
        }
        window.ctx2d.dots = dots;

        setTimeout(function() {
            window.ctx2d.resize();
            window.ctx2d.animate();
        }, 100);

        return window.ctx2d;
    }

    this.canvasCtx2dOptions = canvasCtx2dOptions;
    this.newInstanceCanvs = newInstanceCanvs;
    return this;
})();