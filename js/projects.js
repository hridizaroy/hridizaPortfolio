// Get Experiences data
var experiencesFile = "data/projects.csv";

const Categories = {
    Programming: "Programming",
    Graphics: "Graphics",
    Film: "Film",
    Misc: "Misc"
};

fetch(experiencesFile)
    .then(response => response.text())
    .then(text =>
    {
        var experiencesData = text.toString();

        var data = $.csv.toObjects(experiencesData);

        var accordion = document.createElement("div");
        accordion.className = "custom-accordion";
        accordion.id = "experiencesAccordion";

        var index = 1;

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
                {label:"Technologies", value:element.Technologies},
                {label:"Links", value:element.Links},
                {label:"Type", value:element.Type}
            ];

            var points = eval(element.Points);

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

            // Create Accordion Item
            var accordionID = "accordion" + index;

            var accordionItem = document.createElement("div");
            accordionItem.className = "accordion-item experience";

            var categories = [];

            // TODO: Find a better way to check if these attributes are active
            if (element.Programming == "TRUE")
            {
                categories.push(Categories.Programming);
            }
            if (element.Graphics == "TRUE")
            {
                categories.push(Categories.Graphics);
            }
            if (element.Film == "TRUE")
            {
                categories.push(Categories.Film);
            }
            if (categories.length == 0)
            {
                categories.push(Categories.Misc);
            }

            accordionItem.setAttribute("data-categories", categories);

            var accordionHeading = document.createElement("h2");
            accordionHeading.className = "mb-0";

            var accordionButton = document.createElement("button");
            accordionButton.className = "btn btn-link";
            accordionButton.type = "button";
            accordionButton.setAttribute("data-toggle", "collapse");
            accordionButton.setAttribute("data-target", "#" + accordionID);
            accordionButton.ariaExpanded = "true";
            accordionButton.setAttribute("aria-controls", accordionID);
            accordionButton.innerHTML = element.Title;

            var accordionDataContainer = document.createElement("div");
            accordionDataContainer.id = accordionID;
            accordionDataContainer.className = "collapse show";
            accordionDataContainer.setAttribute("data-parent", "#" + accordion.id);
            accordionDataContainer.setAttribute("aria-labelledBy", "headingOne");

            var accordionData = document.createElement("div");
            accordionData.className = "accordion-body";
            accordionData.appendChild(dataContainerOuter);

            accordionDataContainer.appendChild(accordionData);
            accordionHeading.appendChild(accordionButton);
            
            accordionItem.appendChild(accordionHeading);
            accordionItem.appendChild(accordionDataContainer)

            accordion.appendChild(accordionItem);

            index++;
        });

        // Add experience to portfolio section
        document.querySelector("#portfolio-section .container").appendChild(accordion);
    });

document.querySelectorAll(".filterCheckbox").forEach((elem) =>
{
    elem.addEventListener("change", filterCheckboxHandler);
});

document.querySelector("#orAndToggleCheckbox").addEventListener("change", filterCheckboxHandler);


function filterCheckboxHandler()
{
    var programmingCheckbox = document.getElementById("programmingCheckbox");
    var graphicsCheckbox = document.getElementById("graphicsCheckbox");
    var filmCheckbox = document.getElementById("filmCheckbox");
    var miscCheckbox = document.getElementById("miscCheckbox");

    var orAndToggleCheckbox = document.getElementById("orAndToggleCheckbox");

    var categoryCheckbox = new Object();

    categoryCheckbox[Categories.Programming] = programmingCheckbox;
    categoryCheckbox[Categories.Graphics] = graphicsCheckbox;
    categoryCheckbox[Categories.Film] = filmCheckbox;
    categoryCheckbox[Categories.Misc] = miscCheckbox;

    // if all are unchecked, show all experiences
    if (!programmingCheckbox.checked && !graphicsCheckbox.checked && !filmCheckbox.checked && !miscCheckbox.checked)
    {
        document.querySelectorAll(".experience").forEach(elem =>
            {
                elem.style.display = "block";
            });
    }
    else
    {
        document.querySelectorAll(".experience").forEach(elem =>
            {
                var categories = elem.getAttribute("data-categories");
    
                var categoriesList = categories.split(",");

                var showExperience = false;

                if (orAndToggleCheckbox.checked)
                {
                    showExperience = true;

                    Object.entries(Categories).forEach(categoryPair =>
                        {
                            category = categoryPair[1];
                            if ((categoryCheckbox[category].checked && !categoriesList.includes(category)))
                            {
                                showExperience = false;
                            }
                        });
                }
                else
                {
                    categoriesList.forEach(category =>
                        {
                            if (categoryCheckbox[category].checked)
                            {
                                showExperience = true;
                            }
                        });
                }
    
                if (showExperience)
                {
                    elem.style.display = "block";
                }
                else
                {
                    elem.style.display = "none";
                }
            });
    }
}