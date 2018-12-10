window.Tesseract = Tesseract.create({
    workerPath: 'https://cdn.rawgit.com/naptha/tesseract.js/0.2.0/dist/worker.js',
    corePath: 'https://cdn.rawgit.com/naptha/tesseract.js-core/0.1.0/index.js',
    langPath: 'https://cdn.rawgit.com/naptha/tessdata/gh-pages/3.02/'
});

var processing = false; // Flag to avoid multiple analysis at the same time

$(document).ready(function() {
    $('#start').click(function() {
        if(processing)
            return;

        var file = $('#file-image').prop('files')[0];
        console.log(file);
        processing = true;

        // Tesseract magic
        Tesseract.recognize(file)
            .progress(function(p) {
                // Update progress bar and status
                var progress = Math.floor(p.progress * 100);

                $("#status").text(p.status);
                $("#progress").attr('aria-valuenow', progress);
                $("#progress").width(progress + '%');
            })
            .catch(err => console.error('Error', err))
            .then(function(result) {
                console.log('result', result);
                processing = false; // Another image can be analyzed
                analyze(file, result); // Draw result and predict using TensorFlow

                // On finish update results
                $("#transcripted-text").text(result.text);
                $("#confidence").text(result.confidence + "% of accurracy");
            });
    });

    /**
     * Analyze handwritten symbols using @result from Tesseract
     * @image - Image used
     * @result - Tesseract JSON info(Bounding boxes, separated symbols, confidence per symbol, etc)
     */
    function analyze(image, result) {
        var c = document.getElementById("img-canvas");
        var ctx = c.getContext("2d");
        var reader = new FileReader();

        // Config context
        ctx.lineWidth = 1;

        // Load img object from Base64 data
        reader.readAsDataURL(image);
        reader.onload = function (e) {
            var img = new Image;

            // Draw Tesseract Info on top of the image.
            img.onload = function() {
                ctx.drawImage(img, 0, 0, c.width, c.height);
                var selectedChar = { confidence : 0 };

                // Draw symbols
                for(var i = 0; i < result.symbols.length; i++) {
                    var s = result.symbols[i];

                    // Map coordinates to canvas size
                    s.bbox.x0 = c.width * s.bbox.x0 / img.width;
                    s.bbox.x1 = c.width * s.bbox.x1 / img.width;
                    s.bbox.y0 = c.height * s.bbox.y0 / img.height;
                    s.bbox.y1 = c.height * s.bbox.y1 / img.height;

                    //console.log(s);

                    // Just draw symbols with confidence greater than x
                    if(s.confidence > 0) {
                        ctx.beginPath();
                        ctx.strokeStyle = '#00ff00';
                        ctx.rect(s.bbox.x0 - 2, s.bbox.y0 - 2, s.bbox.x1-s.bbox.x0 + 4, s.bbox.y1-s.bbox.y0 + 4);
                        //ctx.strokeText(Math.round(s.confidence) + "%", s.bbox.x0+1, s.bbox.y0-1);
                        ctx.stroke();
                    }
                    if(s.text.toUpperCase() == 'M' && s.confidence > selectedChar.confidence)
                        selectedChar = s;
                }

                // Extract m and put it in a separated <img>
                var canvasM = document.createElement('canvas');
                var ctxM = canvasM.getContext("2d");
                var imgM = $('#m-result');

                console.log(selectedChar)
                canvasM.width = (selectedChar.bbox.x1-selectedChar.bbox.x0) * 2.5;
                canvasM.height = (selectedChar.bbox.y1-selectedChar.bbox.y0) * 2.5;

                ctxM.drawImage(
                    c,
                    selectedChar.bbox.x0,
                    selectedChar.bbox.y0,
                    canvasM.width / 2.5,
                    canvasM.height / 2.5,
                    0,
                    0,
                    canvasM.width,
                    canvasM.height
                );
                //ctxM.transform(0.2, 0, 0, 0, 0.2, 0, 0);

                imgM.width(canvasM.width);
                imgM.height(canvasM.height);
                imgM.attr("src", canvasM.toDataURL());

                $("#m-result2").width(28);
                $("#m-result2").height(28);

                var imgData = ctxM.getImageData(0, 0, canvasM.width, canvasM.height);

                // Apply filters
                blackAndWhite(imgData);
                imgData = paddingToImage(ctxM, imgData);

                ctxM.putImageData(imgData, 0, 0);
                $("#m-result2").attr("src", canvasM.toDataURL());

                imgData = scaleImageData(ctxM, imgData, 1 / (imgData.width / 28), 1 / (imgData.height / 28));

                predict(imgData);
            };

            img.src = reader.result; // is the data URL because called with readAsDataURL
        }
    }

    async function predict(imageData) {
        const model = await tf.loadModel('/model/model.json');
        var data = reshapeImageData(imageData);
        data = data.map( x => { return x / 255} )
        //console.log(data)

        const prediction = model.predict(tf.tensor2d(data, [1, 784]));
        var probaWorried = prediction.dataSync()[0];
        var probaNotWorried = prediction.dataSync()[1];

        console.log(prediction)

        if(probaWorried > probaNotWorried) {
            $("#worried-res-text").text("Shown by the increasing height of the humps on the m’s. We predicted that this person has a little fear of being ridiculed and tends to worry what others might think when around strangers.")
            $("#worried-res-prob").text(probaWorried.toFixed(4) * 100 + "%")
        } else {
            $("#worried-res-text").text("Shown by the decreasing height of the humps on the m’s. We predicted that this person doesn't tends to worry about what strangers might think about him/she")
            $("#worried-res-prob").text(probaNotWorried.toFixed(4) * 100 + "%")
        }

        prediction.print();
    }

    /**
     * This function will take an array of RGBA pixels
     * and return only the RED pixels
     */
    function reshapeImageData(imageData) {
        var pixels = [];

        for (var i = 0, n = imageData.data.length; i < n; i += 4)
            pixels.push(imageData.data[i]);

        return pixels;
    }

    /**
     * Each side adds a 15% padding to given Image
     */
    function paddingToImage(ctx, imageData) {
        var ratio = imageData.width / imageData.height;
        var newCanvas = $("<canvas>")
            .attr("width", imageData.width)
            .attr("height", Math.ceil(imageData.height * ratio))[0]; //
        var padding = imageData.width / 7;
        var oriW = imageData.width;
        var oriH = Math.ceil(imageData.height * ratio);
        var scaledW = (imageData.width - padding*2) / oriW;
        var scaledH = (imageData.height - padding*2) / oriH;

        console.log("Ratio: " + ratio)
        console.log("size: " + oriW + ", " + oriH)

        imageData = scaleImageData(ctx, imageData, scaledW, scaledH);

        newCanvas.getContext("2d").fillStyle = "black";
        newCanvas.getContext("2d").fillRect(0, 0, oriW, oriH);
        newCanvas.getContext("2d").putImageData(imageData, padding, padding);

        return newCanvas.getContext("2d").getImageData(0, 0, oriW, oriH);
    }

    /**
     * Scale image data
     */
    function scaleImageData(ctx, imageData, scaleWidth, scaleHeight) {
        var newCanvas = $("<canvas>")
            .attr("width", imageData.width)
            .attr("height", imageData.height)[0];

        newCanvas.getContext("2d").putImageData(imageData, 0, 0);
        ctx.scale(scaleWidth, scaleHeight);
        ctx.drawImage(newCanvas, 0, 0)

        return ctx.getImageData(0, 0, imageData.width * scaleWidth, imageData.height * scaleHeight);
    }

    /**
     * Convert imageData to Black and blackWhite
     * https://stackoverflow.com/questions/45152358/best-rgb-combination-to-convert-image-into-black-and-white-threshold
     */
    function blackAndWhite(imageData) {
        var pixels  = imageData.data;

        for (var i = 0, n = pixels.length; i < n; i += 4) {
            let R = pixels[i];
            let G = pixels[i+1];
            let B = pixels[i+2];
            //let lum = R
            let gray = (0.299 * R + 0.587 * G + 0.114 * B)
            if (gray > 140) { //(R > 100 && G > 100 && B > 100){
                pixels[i  ] = 0;        // red
                pixels[i+1] = 0;        // green
                pixels[i+2] = 0;        // blue
            } else{
                pixels[i  ] = 255;        // red
                pixels[i+1] = 255;        // green
                pixels[i+2] = 255;        // blue
            }
        }

        return pixels;
      }
});