const MODEL_URL = 'model_js/model.json';
const SEED_URL = "model_js/neuralhash_128x96_seed1.dat";
let model, seed;
let log_elem, file_upload, input_image;

// utils
function log(message) {
    log_elem.innerText = message;
    console.log(message);
}

console.error = (...data) => {
    alert("error: "+ data.join(" "))
}

function hash_to_hex(h) {
    var bits = "0b" + h.map(e => (e >= 0.5) ? "1" : "0").join("");
    return BigInt(bits).toString(16);
}


document.addEventListener("DOMContentLoaded", function() {
    log_elem = document.getElementById("log");
    file_upload = document.getElementById("input_file");
    input_image = document.getElementById("input_image");


    // upload handler
    file_upload.onchange = function() {
        if(!this.files || !this.files[0]) return;
        input_image.src = URL.createObjectURL(this.files[0]);
    };


    // run the model on the image
    input_image.onload = async () => {
        log("running neuralhash (may take a bit on first run)");
        // have to call twice because js weirdness
        tf.browser.fromPixels(input_image);
        var image = tf.browser.fromPixels(input_image)
            .resizeBilinear([360,360])
            .div(255.0)
            .mul(2.0).sub(1.0)
            .transpose([2,0,1])
            .reshape([1,3,360,360]);

        var model_output = model.execute({"image": image});
        var hash_output = seed.dot(model_output[0].flatten());
        var hash_arr = await hash_output.array();
        var hash_hex = hash_to_hex(hash_arr);
        log("result: "+hash_hex);
    }
});


// load model and seed file
async function load_model() {
    model = await tf.loadGraphModel(MODEL_URL);
    var resp = await fetch(SEED_URL);
    var buff = (await resp.arrayBuffer()).slice(128);
    buff = new Float32Array(buff);
    seed = tf.tensor(buff, [96,128], "float32");
}