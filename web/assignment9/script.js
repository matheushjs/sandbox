
function processInput(event){
  var period = document.querySelector("#period").value;
  var buysell = document.querySelector("#buysell").value;
  var order = document.querySelector("#order").value;

  // Object we will need to send to the bank
  var sendData = {
    dataInicial: null,
    dataFinalCotacao: null,
    $top: 1000,
    $skip: 0,
    $orderby: "cotacaoCompra desc",
    $format: "json",
    $select: "cotacaoCompra,cotacaoVenda,dataHoraCotacao",
  };

  var now = new Date(Date.now());
  var dateBegin = new Date(2019, now.getMonth(), now.getDate()); // Data inicial
  sendData.dataInicial = dateBegin.getMonth() + "-" + dateBegin.getDate() + "-" + dateBegin.getFullYear();

  // 'period' is the number of days
  // We need it to convert in number of milliseconds
  var milli = period * 24 * 60 * 60 * 1000;
  var past = new Date(Date.now() - milli);
  var dateEnd = new Date(2019, past.getMonth(), past.getDate());
  sendData.dataFinalCotacao = dateEnd.getMonth() + "-" + dateEnd.getDate() + "-" + dateEnd.getFullYear();

  if(buysell === "buy" && order === "asc"){
    sendData.$orderby = "cotacaoCompra asc";
  } else if(buysell === "buy" && order === "desc"){
    sendData.$orderby = "cotacaoCompra desc";
  } else if(buysell === "sell" && order === "asc"){
    sendData.$orderby = "cotacaoVenda asc";
  } else if(buysell === "sell" && order === "desc"){
    sendData.$orderby = "cotacaoVenda desc";
  } else {
    console.err("Something went wrong.");
    return;
  }

  if(buysell === "buy"){
    sendData.$select = "cotacaoCompra,dataHoraCotacao";
  } else if(buysell === "sell"){
    sendData.$select = "cotacaoVenda,dataHoraCotacao";
  } else {
    console.err("Something went wrong.");
  }

  console.log(sendData);

  event.preventDefault();
}

window.onload = function(){
  var button = document.querySelector("button");
  button.onclick = processInput;
}
