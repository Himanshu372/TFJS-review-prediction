

let model;

const modelURL = 'http://localhost:5000/model';

const formInput1 = document.getElementById("formInput1");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const form = document.getElementById("container");
const result_span = document.getElementById("number-of-files");
//const numberOfFiles = document.getElementById("number-of-files");
//const fileInput = document.getElementById('file');


const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
    //    const files = fileInput.files;

//    [...files].map(async (img) => {
    const formInput2 = formInput1.value;
    const data = new FormData();
    data.append('formInput1', formInput2);

    const processedImage = await fetch("http://localhost:5000/api/prepare", {
        method: 'POST',
        mode: 'cors',
        body: data
    }).then(response => {
            return response.json();
        }).then(result => {
            console.log('resposne: ', result['ndarray'][0]);
            return tf.tensor(result['ndarray']);
        });
    // shape has to be the same as it was for training of the model

    let prediction = model.predict(processedImage);
//        console.log(${prediction});
    console.log("prediction :- ", prediction);

    res = prediction.dataSync();
    console.log("Prediction array :", res);
    res[0] > res[1] ? res = "Accepted" : res = "Rejected";
    console.log("Final res :", res);
    result_span.innerHTML = res;

return res;
//    })
};


function getTokenisedWord(seedWord) {
  const _token = word2index[seedWord.toLowerCase()]
  return tf.tensor1d([_token])
}

//function showHideDiv(ele) {
//				var srcElement = document.getElementById(ele);
//				if (srcElement != null) {
//					if (srcElement.style.display == "block") {
//						srcElement.style.display = 'none';
//					}
//					else {
//						srcElement.style.display = 'block';
//					}
//					return false;
//				}
//			};


formInput1.addEventListener("blur",() => predict(modelURL));
formInput1.addEventListener("focus",() => result_span.innerHTML = "");
//predictButton.addEventListener("click", () => predict(modelURL));
//clearButton.addEventListener("click", () => {
//    formInput1.value = "";
//    console.log('Clear')
//});

