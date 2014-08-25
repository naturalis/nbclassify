jQuery( document ).ready(function( $ ) {
    // Toggable side navigation.
    (function (document) {

        var layout   = document.getElementById('layout'),
            menu     = document.getElementById('menu'),
            menuLink = document.getElementById('menuLink');

        function toggleClass(element, className) {
            var classes = element.className.split(/\s+/),
                length = classes.length,
                i = 0;

            for(; i < length; i++) {
              if (classes[i] === className) {
                classes.splice(i, 1);
                break;
              }
            }
            // The className is not found
            if (length === classes.length) {
                classes.push(className);
            }

            element.className = classes.join(' ');
        }

        menuLink.onclick = function (e) {
            var active = 'active';

            e.preventDefault();
            toggleClass(layout, active);
            toggleClass(menu, active);
            toggleClass(menuLink, active);
        };

    }(document));

    // Identify Photo button.
    $('#identify_form button').click(function(event) {
        $("#identity").html('<div class="message"><i class="fa fa-gear fa-fw"></i> Please wait while we identify your photo...</div>');
        $("#identify_form").ajaxForm({
            target: '#identity'
        }).submit();

        // Cancel the default submit action.
        event.preventDefault();
    });
});
