jQuery( document ).ready(function( $ ) {
    // Togglable side navigation.
    $( "#menuLink" ).click(function( event ) {
        event.preventDefault();
        $( "#layout" ).toggleClass( "active" );
        $( "#menu" ).toggleClass( "active" );
        $( "#menuLink" ).toggleClass( "active" );
    });

    // Update the badge for the My Photos menu item.
    $.ajax({
        url: "/orchid/session_photo_ids.json",
        dataType: "json",
        success: function(data) {
            $("#my-photos-n").text(data.length);
        }
    });
});
