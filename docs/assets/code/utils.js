function hideAndShow(row, cell, filter, idx) {
    if (typeof(cell) !== 'undefined')
    {
      if (cell.innerHTML.toUpperCase().indexOf(filter) > -1) {
        console.log(cell.innerHTML.toUpperCase(), filter);
        row[idx].style.display = "";
      } else {
        row[idx].style.display = "none";
      }
    }
}

function searchFunc() {
        // Declare variables
        var input, filter, ul, tr, td, a, i, j;
        input = document.getElementById("mySearch");
        filter = input.value.toUpperCase();
        ul = document.getElementById("myMenu");
        tr = ul.getElementsByTagName("tr");
        search_in_file   = false;
        search_in_folder = false;
        search_in_func   = true;
        if (filter.substring(0, 6) == ":FILE:")
        {
          console.log("Searching in file");
          filter = filter.substring(6);
          search_in_file = true;
          search_in_func = false; 
        } else if (filter.substring(0, 8) == ":FOLDER:") 
        { 
          console.log("Searching in folder");
          filter = filter.substring(8);
          search_in_folder = true;
          search_in_func   = false;

        }
        // Loop through all list items, and hide those who don't match the search query
        for (i = 1; i < tr.length; i++) {
          td = tr[i].getElementsByTagName("td");
          for (j = 0; j < td.length; j++) {
            var a = undefined
            if (search_in_file && td[j].id == "file") {
              a = td[j];
            } else if (search_in_folder && td[j].id == "folder") {
              a = td[j];
            } else if (search_in_func && td[j].id == "func_name") {
              a = td[j];
            }
            hideAndShow(tr, a, filter, i)
        }
      }
}