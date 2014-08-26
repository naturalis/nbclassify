jQuery( document ).ready(function( $ ) {
    // Toggable side navigation.
    $( "#menuLink" ).click(function( event ) {
        event.preventDefault();
        $( "#layout" ).toggleClass( "active" );
        $( "#menu" ).toggleClass( "active" );
        $( "#menuLink" ).toggleClass( "active" );
    });
});
