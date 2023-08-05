// Get Experiences data
var experiencesFile = "data/experiences.csv";

fetch(experiencesFile)
    .then(response => response.text())
    .then(text =>
    {
        var experiencesData = text.toString();

        var data = $.csv.toObjects(experiencesData);

        // Iterate through experiences and create html elements
        data.forEach(element =>
        {
            var layer1 = document.createElement("div");
            layer1.className = "portfolio-single-wrap unslate_co--section";

            var layer2 = document.createElement("div");
            layer2.className = "portfolio-single-inner";

            var heading = document.createElement("h2");
            heading.className = "heading-portfolio-single-h3 text-black gsap-reveal-hero pb-5";
            heading.innerHTML = element.Title;

            var dataContainerOuter = document.createElement("div");
            dataContainerOuter.className = "row mb-5 align-items-stretch";

            var dataContainerInner = document.createElement("div");
            dataContainerInner.className = "col-lg-12 pl-lg-5";

            var dataRow = document.createElement("div");
            dataRow.className = "row mb-3";

            // Store columns in a list for easier iteration
            var cols = [
                {label:"Date", value:element.Date},
                {label:"Employer", value: "<a href = \"#\">" + element.Employer + "</a>"},
                {label:"Department", value:element.Department},
                {label:"Location", value:element.Location}
            ];

            var points = [element.Point_1, element.Point_2, element.Point_3];

            cols.forEach(colData =>
            {
                var column = document.createElement("div");
                column.className = "col-sm-6 col-md-6 col-lg-6 mb-4";

                var colDataContainer = document.createElement("div");
                colDataContainer.className = "detail-v1";
            
                var label = document.createElement("span");
                label.className = "detail-label";
                label.innerHTML = colData.label;
            
                var value = document.createElement("span");
                value.className = "detail-val";
                value.innerHTML = colData.value;

                colDataContainer.appendChild(label);
                colDataContainer.appendChild(value);

                column.appendChild(colDataContainer);

                dataRow.appendChild(column);
            });

            var details = document.createElement("ul");
            points.forEach(point =>
            {
                if (point)
                {
                    var pointLi = document.createElement("li");
                    pointLi.innerHTML = point;
    
                    details.appendChild(pointLi);
                }
            });

            dataContainerInner.appendChild(dataRow);
            dataContainerInner.appendChild(details);

            dataContainerOuter.appendChild(dataContainerInner);

            layer2.appendChild(heading);
            layer2.appendChild(dataContainerOuter);
            layer1.appendChild(layer2);

            // Add experience to portfolio section
            document.querySelector("#portfolio-section .container").appendChild(layer1);
        });
    });