<html>
    <head>
        <meta charset="utf-8">
        <title>Emotion choropleth</title>
        <script src="{{url_for('static',filename='d3.min.js')}}"></script>
		<script src="{{url_for('static',filename='topojson.min.js')}}"></script>
		<script src="{{url_for('static',filename='datamaps.world.min.js')}}"></script>
    <script src="https://code.jquery.com/jquery-3.1.1.min.js"></script>
    <style>

          g.legend{
            opacity: .8;
          }
          body{
            background: linear-gradient(to right, rgba(183,222,237,1) 0%, rgba(113,206,239,1) 27%, rgba(33,180,226,1) 54%, rgba(167,198,219,1) 100%);
            background-color: white;
            font-family: Lato, "Open Sans";
          }
          h1{
            text-align: center;
            color: red;
            text-shadow: 2px 2px 5px #000000;
          }
          strong{
            color: #FF7F00;

          }
          svg{

            padding: 30px 30px 30px 30px;
            border: 2px solid black;
            border-style: outset;
          }
      </style>

</head>
    <body>

        Key Word  <input type="text" name="" id="key" placeholder="Enter A Key Word"><br>
        Start date <input type="text" name="" id="start_date"><br>
        End date <input type="text" name="" id="end_date"><br>
        Max tweets<input type="text" name="" id="max" required><br>
        <input type="submit" name="submit" onclick="start()" value="check">
        <input type="reset" value="reset">

    	<h1 >ORBIS <strong>SPECTRA</strong></h1>
    <div id="container" style=" width:94%; height:80%; "></div>
    	<script>
        function start(){
          call_me()
        }
        function call_me(){
          var key = document.getElementById("key").value
            var start_date = document.getElementById("start_date").value
              var end_date = document.getElementById("end_date").value
                var max = document.getElementById("max").value
            $.ajax({
              type: 'POST',
              url: '/timeline',
              dataType: 'json',
              contentType: "application/json;charset=UTF-8",
              data : JSON.stringify({'key': key, 'start_date': start_date, 'end_date': end_date, 'max':max} ),
              success: function(response){
                $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
                map.updateChoropleth(response)
              },
              error: function(error){
                console.log(error);
              }
            });
        }

    		var map = new Datamap({element: document.getElementById('container'),geographyConfig:{borderColor:'black'},
    		fills:{defaultFill: 'lightgray'}});
    		map.legend();



		</script>
	</body>
</html>
